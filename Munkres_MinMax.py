"""
This code is an implementation of Munkres algorithm https://software.clapper.org/munkres/ 
modified by Simon Ferrier (simon.ferrier@inria.fr) to minimize the maximal cost, instead of the total cost.
"""

__docformat__ = 'markdown'

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import copy
from typing import Union, NewType, Sequence, Tuple, Optional, Callable

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__     = ['MunkresMinMax', 'make_cost_matrix', 'DISALLOWED']

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

AnyNum = NewType('AnyNum', Union[int, float])
Matrix = NewType('Matrix', Sequence[Sequence[AnyNum]])

# Constants
class DISALLOWED_OBJ(object):
    pass
DISALLOWED = DISALLOWED_OBJ()
DISALLOWED_PRINTVAL = "D"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsolvableMatrix(Exception):
    """
    Exception raised for unsolvable matrices
    """
    pass

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class MunkresMinMax:
    """
    Calculate the Munkres solution to the classical assignment problem.
    See the module documentation for usage.
    """

    def __init__(self):
        """Create a new instance"""
        self.C = None
        self.last_makespan=None
        self.makespan=None
        self.bin_C=None
        self.row_covered = []
        self.col_covered = []
        self.n = 0
        self.Z0_r = 0
        self.Z0_c = 0
        self.last_marked=None
        self.marked = None
        self.path = None

    def pad_matrix(self, matrix: Matrix, pad_value: int=0) -> Matrix:
        """
        Pad a possibly non-square matrix to make it square.

        **Parameters**

        - `matrix` (list of lists of numbers): matrix to pad
        - `pad_value` (`int`): value to use to pad the matrix

        **Returns**

        a new, possibly padded, matrix
        """
        max_columns = 0
        total_rows = len(matrix)

        for row in matrix:
            max_columns = max(max_columns, len(row))

        total_rows = max(max_columns, total_rows)

        new_matrix = []
        for row in matrix:
            row_len = len(row)
            new_row = row[:]
            if total_rows > row_len:
                # Row too short. Pad it.
                new_row += [pad_value] * (total_rows - row_len)
            new_matrix += [new_row]

        while len(new_matrix) < total_rows:
            new_matrix += [[pad_value] * total_rows]

        return new_matrix

    def compute(self, cost_matrix: Matrix) -> Sequence[Tuple[int, int]]:
        """
        Compute the indexes for the lowest-cost pairings between rows and
        columns in the database. Returns a list of `(row, column)` tuples
        that can be used to traverse the matrix.

        **WARNING**: This code handles square and rectangular matrices. It
        does *not* handle irregular matrices.

        **Parameters**

        - `cost_matrix` (list of lists of numbers): The cost matrix. If this
          cost matrix is not square, it will be padded with zeros, via a call
          to `pad_matrix()`. (This method does *not* modify the caller's
          matrix. It operates on a copy of the matrix.)


        **Returns**

        A list of `(row, column)` tuples that describe the lowest cost path
        through the matrix
        """
        self.C = self.pad_matrix(cost_matrix)
        self.C_bool=[]
        self.n = len(self.C)
        self.original_length = len(cost_matrix)
        self.original_width = len(cost_matrix[0])
        self.row_covered = [False for i in range(self.n)]
        self.col_covered = [False for i in range(self.n)]
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = self.__make_matrix(self.n * 2, 0)

        self.last_marked=self.__make_matrix(self.n, 0)
        self.last_makespan=None
        self.marked = self.__make_matrix(self.n, 0)
        self.bin_C=self.__make_matrix(self.n, 0)

        done = False
        step = 1

        steps = { 1 : self.__step1,
                  2 : self.__step2,
                  3 : self.__step3,
                  4 : self.__step4,
                  5 : self.__step5,
                  6 : self.__step6 }

        while not done:
            try:
                func = steps[step]
                step = func()
            except KeyError:
                done = True

        # Look for the starred columns
        results = []
        for i in range(self.original_length):
            for j in range(self.original_width):
                if self.last_marked[i][j] == 1:
                    results += [(i, j)]

        return results, self.last_makespan

    def __copy_matrix(self, matrix: Matrix) -> Matrix:
        """Return an exact copy of the supplied matrix"""
        return copy.deepcopy(matrix)

    def __make_matrix(self, n: int, val: AnyNum) -> Matrix:
        """Create an *n*x*n* matrix, populating it with the specific value."""
        matrix = []
        for i in range(n):
            matrix += [[val for j in range(n)]]
        return matrix

    def __step1(self) -> int:
        """
        Mark the diagonal. Find the makespan. Set all value >= the Makespan to 1 and others to 0.
        """
        # print("Step 1")

        C = self.C
        n = self.n
        makespan=0
        for i in range(n):
            self.last_marked[i][i]=1
            if self.C[i][i]>makespan:
                makespan=self.C[i][i]
        self.last_makespan=makespan
        for i in range(n):
            for j in range(n):
                if C[i][j]>=makespan:
                    self.bin_C[i][j]=1

        return 2

    def __step2(self) -> int:
        # print("Step 2")
        # print(self.bin_C)
        """
        Find a zero (Z) in the resulting matrix. If there is no starred
        zero in its row or column, star Z. Repeat for each element in the
        matrix. Go to Step 3.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                if (self.bin_C[i][j] == 0) and \
                        (not self.col_covered[j]) and \
                        (not self.row_covered[i]):
                    self.marked[i][j] = 1
                    self.col_covered[j] = True
                    self.row_covered[i] = True
                    break

        self.__clear_covers()
        # print(self.marked)
        return 3

    def __step3(self) -> int:
        # print("Step 3")
        # print(self.bin_C)
        # print(self.marked)
        # print(self.last_marked)
        """
        Cover each column containing a starred zero. If K columns are
        covered, the starred zeros describe a complete set of unique
        assignments. In this case, Go to step 6, otherwise, Go to Step 4.
        """
        n = self.n
        count = 0
        makespan=0
        for i in range(n):
            for j in range(n):
                if self.marked[i][j] == 1 :
                    if self.C[i][j]>makespan:
                        makespan=self.C[i][j]
                    if not self.col_covered[j]:
                        self.col_covered[j] = True
                        count += 1

        if count >= n:
            self.__save_marked()
            self.makespan=makespan
            self.last_makespan=makespan
            step = 6
        else:
            step = 4

        return step

    def __step4(self) -> int:
        # print("Step 4")

        """
        Find a noncovered zero and prime it. If there is no starred zero
        in the row containing this primed zero, Go to Step 5. Otherwise,
        cover this row and uncover the column containing the starred
        zero. Continue in this manner until there are no uncovered zeros
        left. If no possible solution, go to DONE.
        """
        step = 0
        done = False
        row = 0
        col = 0
        star_col = -1
        while not done:
            (row, col) = self.__find_a_zero(row, col)
            if row < 0:
                done = True
                step = 7 # done
            else:
                self.marked[row][col] = 2
                star_col = self.__find_star_in_row(row)
                if star_col >= 0:
                    col = star_col
                    self.row_covered[row] = True
                    self.col_covered[col] = False
                else:
                    done = True
                    self.Z0_r = row
                    self.Z0_c = col
                    step = 5

        return step

    def __step5(self) -> int:
        # print("Step 5")

        """
        Construct a series of alternating primed and starred zeros as
        follows. Let Z0 represent the uncovered primed zero found in Step 4.
        Let Z1 denote the starred zero in the column of Z0 (if any).
        Let Z2 denote the primed zero in the row of Z1 (there will always
        be one). Continue until the series terminates at a primed zero
        that has no starred zero in its column. Unstar each starred zero
        of the series, star each primed zero of the series, erase all
        primes and uncover every line in the matrix. Return to Step 3
        """
        count = 0
        path = self.path
        path[count][0] = self.Z0_r
        path[count][1] = self.Z0_c
        done = False
        while not done:
            row = self.__find_star_in_col(path[count][1])
            if row >= 0:
                count += 1
                path[count][0] = row
                path[count][1] = path[count-1][1]
            else:
                done = True

            if not done:
                col = self.__find_prime_in_row(path[count][0])
                count += 1
                path[count][0] = path[count-1][0]
                path[count][1] = col

        self.__convert_path(path, count)
        self.__clear_covers()
        self.__erase_primes()
        return 3

    def __step6(self) -> int:
        # print("Step 6")

        """
        Update the binary C with new makespan. Unmark all Zeros and go back to step 2.
        """
        
        for i in range(self.n):
            for j in range(self.n):
                self.marked[i][j]=0
                if self.C[i][j]>=self.makespan:
                    self.bin_C[i][j]=1
                
        return 2

    def __find_smallest(self) -> AnyNum:
        """Find the smallest uncovered value in the matrix."""
        minval = sys.maxsize
        for i in range(self.n):
            for j in range(self.n):
                if (not self.row_covered[i]) and (not self.col_covered[j]):
                    if self.C[i][j] is not DISALLOWED and minval > self.C[i][j]:
                        minval = self.C[i][j]
        return minval


    def __find_a_zero(self, i0: int = 0, j0: int = 0) -> Tuple[int, int]:
        """Find the first uncovered element with value 0"""
        row = -1
        col = -1
        i = i0
        n = self.n
        done = False

        while not done:
            j = j0
            while True:
                if (self.bin_C[i][j] == 0) and \
                        (not self.row_covered[i]) and \
                        (not self.col_covered[j]):
                    row = i
                    col = j
                    done = True
                j = (j + 1) % n
                if j == j0:
                    break
            i = (i + 1) % n
            if i == i0:
                done = True

        return (row, col)

    def __find_star_in_row(self, row: Sequence[AnyNum]) -> int:
        """
        Find the first starred element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 1:
                col = j
                break

        return col

    def __find_star_in_col(self, col: Sequence[AnyNum]) -> int:
        """
        Find the first starred element in the specified row. Returns
        the row index, or -1 if no starred element was found.
        """
        row = -1
        for i in range(self.n):
            if self.marked[i][col] == 1:
                row = i
                break

        return row

    def __find_prime_in_row(self, row) -> int:
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = -1
        for j in range(self.n):
            if self.marked[row][j] == 2:
                col = j
                break

        return col

    def __convert_path(self,
                       path: Sequence[Sequence[int]],
                       count: int) -> None:
        
        for i in range(count+1):
            if self.marked[path[i][0]][path[i][1]] == 1:
                self.marked[path[i][0]][path[i][1]] = 0
            else:
                self.marked[path[i][0]][path[i][1]] = 1
                # if self.C[path[i][0]][path[i][1]]>makespan:
                #     makespan=self.C[path[i][0]][path[i][1]]
        # for i in range(self.n):
        #     for j in range(self.n):
        #         if self.C[i][j]>=makespan:
        #             self.bin_C[i][j]=1
                

    def __clear_covers(self) -> None:
        """Clear all covered matrix cells"""
        for i in range(self.n):
            self.row_covered[i] = False
            self.col_covered[i] = False

    def __erase_primes(self) -> None:
        """Erase all prime markings"""
        for i in range(self.n):
            for j in range(self.n):
                if self.marked[i][j] == 2:
                    self.marked[i][j] = 0
    def __save_marked(self) -> None:
        for i in range(self.n):
            for j in range(self.n):
                self.last_marked[i][j]=self.marked[i][j]

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def make_cost_matrix(
        profit_matrix: Matrix,
        inversion_function: Optional[Callable[[AnyNum], AnyNum]] = None
    ) -> Matrix:
    """
    Create a cost matrix from a profit matrix by calling `inversion_function()`
    to invert each value. The inversion function must take one numeric argument
    (of any type) and return another numeric argument which is presumed to be
    the cost inverse of the original profit value. If the inversion function
    is not provided, a given cell's inverted value is calculated as
    `max(matrix) - value`.

    This is a static method. Call it like this:

        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, inversion_func)

    For example:

        from munkres import Munkres
        cost_matrix = Munkres.make_cost_matrix(matrix, lambda x : sys.maxsize - x)

    **Parameters**

    - `profit_matrix` (list of lists of numbers): The matrix to convert from
       profit to cost values.
    - `inversion_function` (`function`): The function to use to invert each
       entry in the profit matrix.

    **Returns**

    A new matrix representing the inversion of `profix_matrix`.
    """
    if not inversion_function:
      maximum = max(max(row) for row in profit_matrix)
      inversion_function = lambda x: maximum - x

    cost_matrix = []
    for row in profit_matrix:
        cost_matrix.append([inversion_function(value) for value in row])
    return cost_matrix

def print_matrix(matrix: Matrix, msg: Optional[str] = None) -> None:
    """
    Convenience function: Displays the contents of a matrix.

    **Parameters**

    - `matrix` (list of lists of numbers): The matrix to print
    - `msg` (`str`): Optional message to print before displaying the matrix
    """
    import math

    if msg is not None:
        print(msg)

    # Calculate the appropriate format width.
    width = 0
    for row in matrix:
        for val in row:
            if val is DISALLOWED:
                val = DISALLOWED_PRINTVAL
            width = max(width, len(str(val)))

    # Make the format string
    format = ('%%%d' % width)

    # Print the matrix
    for row in matrix:
        sep = '['
        for val in row:
            if val is DISALLOWED:
                val = DISALLOWED_PRINTVAL
            formatted = ((format + 's') % val)
            sys.stdout.write(sep + formatted)
            sep = ', '
        sys.stdout.write(']\n')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

        
    # matrix = [[1,0,0,1,1],
    #         [0,1,1,0,1],
    #         [0,0,0,1,1],
    #         [1,0,0,1,1],
    #         [1,0,1,0,0]]
    matrix = [[4,9,3,3,8],[1,7,2,5,9],[9,8,3,1,1],[8,5,4,6,2],[2,5,7,9,4]]
    m = MunkresMinMax()
    indexes, makespan = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    max = 0
    for row, column in indexes:
        value = matrix[row][column]
        print(f'({row}, {column}) -> {value}')
    print(f'Makespan : {makespan}')