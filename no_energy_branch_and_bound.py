from numpy import inf as INF
import matplotlib.pyplot as plt
import random
import time
from queue import PriorityQueue
from Node import Node
from no_energy_heuristic import heuristic
from matplotlib import patches, lines
from field_and_rows import *
from Utils import LineSeg, Agent, Row, draw_semicircle 

colors=['b','g','r','c','m','y']

plt.rcParams.update({'font.size': 22})
class OptimalSearch():
    def __init__(self, line_pts, agents_list, timeout_time, h_makespan_slow_depth, h_prio_mode=1):


        self.timeout_time=timeout_time
        self.line_pts=line_pts

        linesegs = [LineSeg(line_pts[i][0], line_pts[i][1]) for i in range(len(line_pts))]
        self.lengths = [lineseg.length() for lineseg in linesegs]

        self.rows_list=[]
        self.fill_rows_list()

        self.agents_list=agents_list
        self.nb_nodes=1
        self.prioQ: PriorityQueue[Node] = PriorityQueue()
        self.upper_bound=INF
        self.best_allocation=None

        self.h_makespan_slow_depth = h_makespan_slow_depth 
        self.h_prio_mode=h_prio_mode # 1 : h_prio everytime, 2 : 0 when leaf


    def get_optimal_solution(self):
         return self.best_allocation, self.upper_bound

    def fill_rows_list(self):
            id_row=0
            for r in self.line_pts:
                row=Row(id_row, r, self.lengths[id_row], self.lengths[id_row])
                id_row+=1
                self.rows_list.append(row)
                

    def find_optimal(self):

        nb_agents=len(self.agents_list)
        starting_points = [ag.get_starting_point() for ag in self.agents_list]

        self.best_allocation, self.upper_bound=heuristic(self.agents_list, line_pts,starting_points)
        print("Heuristic makespan : ", self.upper_bound)
        first_node=Node(0, 0, None, None, 0)
        self.prioQ.put(first_node)
        average=sum([row.length for row in self.rows_list])/nb_agents

        if self.timeout_time==None:
            while not self.prioQ.empty():
                node_to_open=self.prioQ.get()

                res=node_to_open.open(self.rows_list, self.agents_list, self.nb_nodes, self.upper_bound, average, self.prioQ, self.h_makespan_slow_depth, self.h_prio_mode)

                if res!=1 and res[1]<=self.upper_bound:
                    self.upper_bound=res[1]
                    self.best_allocation=res[0]
            return 1

        else:
            while not self.prioQ.empty() and time.time()<self.timeout_time:
                node_to_open=self.prioQ.get()

                res=node_to_open.open(self.rows_list, self.agents_list, self.nb_nodes, self.upper_bound, average, self.prioQ, self.h_makespan_slow_depth, self.h_prio_mode)

                if res!=1 and res[1]<=self.upper_bound:
                    self.upper_bound=res[1]
                    self.best_allocation=res[0]
                    # print("New best makespan found : ", self.upper_bound)

            for allocation in self.best_allocation:
                allocation[0].set_rows(allocation[1])
                allocation[0].set_entry_point(allocation[2], allocation[3])
            
            if time.time()<self.timeout_time:
                is_opti=True
            else:
                is_opti=False
            
            return is_opti
            
        # print("Optimal solution found ! ")
       
### Parameters
NB_SIDE_POLYGON=8
NB_ROWS=50
NB_AGENTS=4
###

if __name__ == "__main__":

    rd_seed=15
    random.seed(rd_seed)
    points = to_convex_contour(NB_SIDE_POLYGON, rd_seed)
    line_pts=rows_creator_nb_rows(points,NB_ROWS)
    starting_points = random_starting_points(NB_AGENTS, rd_seed)

    id=0
    agents_list=[]
    for sp in starting_points:
        ag=Agent(id, sp, None, None, None, None, None)
        agents_list.append(ag)
        id+=1

    before=time.time()
    optimal_search=OptimalSearch(line_pts, agents_list, None,5, 1)
    is_opti=optimal_search.find_optimal()
    optimal_alloaction, makespan=optimal_search.get_optimal_solution()
    after=time.time()

    print("Branch and Bound makespan : ", makespan)
    print("Time (in seconds) to find solution : ",after-before)
    ## We create the final fig ##
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Branch and Bound result')
    polygon = patches.Polygon(points, closed = True, fill = False)
    ax.add_artist(polygon)
    for ag in optimal_alloaction :

        color = colors[ag.id]
        ax.plot(ag.starting_point[0], ag.starting_point[1], marker="X", markersize=15, color=color)
        ind=0
        for stage in ag.actions:
            
            if stage.type_of_travel==0 or stage.type_of_travel==1:
                ax.add_artist(lines.Line2D([stage.start_pt[0], stage.end_pt[0]],[stage.start_pt[1], stage.end_pt[1]], color=color))
            elif stage.type_of_travel==2:
                x1, y1 = stage.start_pt[0], stage.start_pt[1]
                x2, y2 = stage.end_pt[0], stage.end_pt[1]
                if (ag.ind_entry_p==0) and min(ag.rows, key=lambda r : r.get_id())==ag.rows[0] or (ag.ind_entry_p==1) and min(ag.rows, key=lambda r : r.get_id())==ag.rows[-1]:
                    if ind%2==0:
                        draw_semicircle(x1, y1, x2, y2, color=color, lw=1, ax=ax)
                    else:
                        draw_semicircle(x2, y2, x1, y1, color=color, lw=1, ax=ax)
                else:
                    if ind%2==1:
                        draw_semicircle(x1, y1, x2, y2, color=color, lw=1, ax=ax)
                    else:
                        draw_semicircle(x2, y2, x1, y1, color=color, lw=1, ax=ax)
                ind+=1

    plt.show()