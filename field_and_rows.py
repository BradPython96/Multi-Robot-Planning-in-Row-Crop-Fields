import numpy as np
import matplotlib.animation as animation
import random
from math import sin,atan,atan2
from Utils import LineSeg
from copy import copy

ROWS_SPACING=0.25
def random_starting_points(nb_robots, rd_seed):
    random.seed(rd_seed)
    i = 0
    starting_points=[]
    while i<nb_robots:
        x_rand=random.uniform(-0.5, 5.5)
        y_rand=random.uniform(-0.5, 5.5)
        if (x_rand<0 or x_rand>5) and (y_rand<0 or y_rand>5):
            starting_points.append([x_rand,y_rand])
            i+=1
    return starting_points

def to_convex_contour(vertices_count,
                      rd_seed):
    
    random.seed(rd_seed)

    xs = [random.random()*5 for _ in range(vertices_count)]
    ys = [random.random()*5 for _ in range(vertices_count)]

    xs = sorted(xs)
    ys = sorted(ys)

    min_x, *xs, max_x = xs
    min_y, *ys, max_y = ys
    vectors_xs = _to_vectors_coordinates(xs, min_x, max_x)
    vectors_ys = _to_vectors_coordinates(ys, min_y, max_y)
    random.shuffle(vectors_ys)

    def to_vector_angle(vector):
        x, y = vector
        return atan2(y, x)

    vectors = sorted(zip(vectors_xs, vectors_ys),
                     key=to_vector_angle)
    point_x = point_y = 0
    min_polygon_x = min_polygon_y = 0
    points = []
    for vector_x, vector_y in vectors:
        points.append((point_x, point_y))
        point_x += vector_x
        point_y += vector_y
        min_polygon_x = min(min_polygon_x, point_x)
        min_polygon_y = min(min_polygon_y, point_y)
    shift_x, shift_y = min_x - min_polygon_x, min_y - min_polygon_y
    return [(point_x + shift_x, point_y + shift_y)
            for point_x, point_y in points]

def _to_vectors_coordinates(coordinates, min_coordinate, max_coordinate):
    last_min = last_max = min_coordinate
    result = []
    for coordinate in coordinates:
        if _to_random_boolean():
            result.append(coordinate - last_min)
            last_min = coordinate
        else:
            result.append(last_max - coordinate)
            last_max = coordinate
    result.extend((max_coordinate - last_min,
                   last_max - max_coordinate))
    return result

def _to_random_boolean():
    return random.getrandbits(1)

def rows_creator(points):

    linesegs = [LineSeg(points[i], points[i+1]) if i+1 < len(points) else LineSeg(points[i], points[0]) for i in range(len(points))]
    lengths = [lineseg.length() for lineseg in linesegs]
    longest_seg = [lineseg for lineseg in linesegs if lineseg.length() == max(lengths)]
    m = longest_seg[0].m
    b = longest_seg[0].b

    intercept_ranges = [lineseg.intercept_range(m) for lineseg in linesegs]

    max_intercept = np.max(intercept_ranges)
    min_intercept = np.min(intercept_ranges)
    spacing = abs(ROWS_SPACING/(sin(atan(1/m))))
    intercepts = np.arange(min_intercept + spacing, max_intercept, spacing)

    line_pts = [[lineseg.intersect_w_line(m, intercept) for lineseg in linesegs if lineseg.intersect_w_line(m, intercept)[0] is not None] for intercept in intercepts]

    if line_pts[0][0][0]==line_pts[0][1][0]:
        x_equ=True
    else:
        x_equ=False
        
    for line in line_pts:
        if not x_equ:
            if line[0][0]<line[1][0]:
                line.reverse()
        else:
            if line[0][1]<line[1][1]:
                line.reverse()

    return(line_pts)
    
def rows_creator_nb_rows(points, nb_rows):

    linesegs = [LineSeg(points[i], points[i+1]) if i+1 < len(points) else LineSeg(points[i], points[0]) for i in range(len(points))]
    lengths = [lineseg.length() for lineseg in linesegs]
    longest_seg = [lineseg for lineseg in linesegs if lineseg.length() == max(lengths)]
    m = longest_seg[0].m
    b = longest_seg[0].b

    intercept_ranges = [lineseg.intercept_range(m) for lineseg in linesegs]

    max_intercept = np.max(intercept_ranges)
    min_intercept = np.min(intercept_ranges)

    num_lines = nb_rows

    spacing = (max_intercept - min_intercept) / (num_lines+1)
    intercepts = np.arange(min_intercept + spacing, max_intercept, spacing)

    line_pts = [[lineseg.intersect_w_line(m, intercept) for lineseg in linesegs if lineseg.intersect_w_line(m, intercept)[0] is not None] for intercept in intercepts]
    
    while len(line_pts)>nb_rows:
        line_pts.pop()


    if line_pts[0][0][0]==line_pts[0][1][0]:
        x_equ=True
    else:
        x_equ=False
        
    for line in line_pts:
        if not x_equ:
            if line[0][0]<line[1][0]:
                line.reverse()
        else:
            if line[0][1]<line[1][1]:
                line.reverse()

    return(line_pts)

# Vector utility functions
def vec_unit(v):
    length = np.sqrt(v[0] ** 2 + v[1] ** 2)
    return np.array([v[0] / length, v[1] / length])

def vec_mul(v, s):
    return np.array([v[0] * s, v[1] * s])

def vec_dot(v1, v2):
    return np.dot(v1, v2)

def vec_rot_90_cw(v):
    return np.array([v[1], -v[0]])

def vec_rot_90_ccw(v):
    return np.array([-v[1], v[0]])

def intersect(line1, line2):
    a1 = line1[1][0] - line1[0][0]
    b1 = line2[0][0] - line2[1][0]
    c1 = line2[0][0] - line1[0][0]

    a2 = line1[1][1] - line1[0][1]
    b2 = line2[0][1] - line2[1][1]
    c2 = line2[0][1] - line1[0][1]

    t = (b1 * c2 - b2 * c1) / (a2 * b1 - a1 * b2)
    
    return np.array([line1[0][0] + t * (line1[1][0] - line1[0][0]),
                     line1[0][1] + t * (line1[1][1] - line1[0][1])])

def poly_is_cw(p):
    return vec_dot(vec_rot_90_cw(p[1] - p[0]), p[2] - p[1]) >= 0

def draw_polygon(p, ax, color="black"):
    p = np.append(p, [p[0]], axis=0)  # Close the polygon
    ax.plot(p[:, 0], p[:, 1], color)
    for point in p:
        ax.plot(point[0], point[1], marker="o", color="grey", linewidth=0.5)
