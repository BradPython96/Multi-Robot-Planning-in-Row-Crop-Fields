import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from field_and_rows import *
from Utils import LineSeg, give_row, compute_list_of_agents_makespan, compute_makespan_hungarian, Agent, Row, draw_semicircle 
from copy import deepcopy


colors=['b','g','r','c','m','y']


def simple_rows_allocation(line_pts, nb_robots):



    linesegs = [LineSeg(line_pts[i][0], line_pts[i][1]) for i in range(len(line_pts))]

    lengths = [lineseg.length() for lineseg in linesegs]
    total_lenght = sum(lengths)
    length_to_go=total_lenght

    length_per_robot=total_lenght/nb_robots
    robot_cover_length=length_per_robot

    rows_to_cover=[[] for i in range(nb_robots)]
    robot_lenght_to_cover=[0 for i in range(nb_robots)]
    i_robot=0
    

    for row in range(len(lengths)):
        
        row_len=lengths[row]
    
        if i_robot==nb_robots-1:
            rows_to_cover[i_robot].append(row)
            robot_lenght_to_cover[i_robot]+=row_len
            

        else:            
            if robot_cover_length>row_len:
                rows_to_cover[i_robot].append(row)
                robot_lenght_to_cover[i_robot]+=row_len
                robot_cover_length-=row_len
                length_to_go-=row_len

            elif robot_cover_length>row_len/2:
                rows_to_cover[i_robot].append(row)
                robot_lenght_to_cover[i_robot]+=row_len
                i_robot+=1
                length_to_go-=row_len
                robot_cover_length=length_to_go/(nb_robots-i_robot)

            else:
                i_robot+=1
                rows_to_cover[i_robot].append(row)
                robot_lenght_to_cover[i_robot]+=row_len
                robot_cover_length=length_to_go/(nb_robots-i_robot)-row_len
                length_to_go-=row_len

    return rows_to_cover, robot_lenght_to_cover, lengths

def heuristic(agents,line_pts, starting_points):

    simple_blocks, robot_lenght_to_cover, lengths = simple_rows_allocation(line_pts, len(starting_points))
    rows=[]
    id_row=0
    for r in line_pts:
        row=Row(id_row, r, lengths[id_row], lengths[id_row])
        
        rows.append(row)
        id_row+=1

    block_list=[]
    for b in simple_blocks:
        rows_block=[]
        for irow in b:
            rows_block.append(rows[irow])
        block_list.append(rows_block)

    minimal_robot_allocation, minimal_makespan=compute_makespan_hungarian(block_list, agents)

    for alloc in minimal_robot_allocation:
        alloc[0].rows=alloc[1]
        alloc[0].non_ordered_rows=sorted(copy(alloc[1]), key=lambda x: x.get_id())
        alloc[0].compute_entry_point()
        alloc[0].compute_rows_order()
        alloc[0].create_actions()

    ms=0
    init_worst_agent_ind=0
    first_working_agent=0
    for ag in agents :
        
        ag_length=ag.compute_makespan()
        if ag_length>ms:
            ms=ag_length
            init_worst_agent_ind=ag.get_id()
        if ag_length==0:
            agents.insert(0, agents.pop(ag.get_id))
            first_working_agent+=1
            init_worst_agent_ind+=1

    agents=agents[:first_working_agent]+sorted(agents[first_working_agent:], key=lambda agent: agent.rows[0].get_id())
    optimum_found=False

    best_agents_allocation=deepcopy(agents)
    best_makespan=ms

    worst_agent_ind=init_worst_agent_ind
    while not optimum_found:

        side=0 # 0 for left, 1 for right
        new_best_makespan_found=False
        while not new_best_makespan_found:

            if worst_agent_ind>=len(agents)-1 and side==1:
                optimum_found=True
                break
            elif worst_agent_ind==0 and side==0:
                side=1
                agents=deepcopy(best_agents_allocation)
                worst_agent_ind=init_worst_agent_ind
            else:
                give_row(agents, worst_agent_ind, side)
                makespan = compute_list_of_agents_makespan(agents)[1]
                if makespan<best_makespan:
                    best_makespan=makespan
                    best_agents_allocation=deepcopy(agents)
                    new_best_makespan_found=True
                else:
                    if side==0:
                        worst_agent_ind-=1
                    elif side==1:
                        worst_agent_ind+=1

    return best_agents_allocation, best_makespan


if __name__ == "__main__":

    NB_SIDE_POLYGON=8
    NB_ROWS=50
    NB_ROBOTS=4
    rd_seed=1

    points = to_convex_contour(NB_SIDE_POLYGON,rd_seed)
    line_pts=rows_creator_nb_rows(points,NB_ROWS)

    starting_points = random_starting_points(NB_ROBOTS, rd_seed)
    id=0
    agents_list=[]
    for sp in starting_points:
        ag=Agent(id, sp, None, None, None, None, None)
        agents_list.append(ag)
        id+=1    
    agents_allocation, makespan = heuristic(agents_list,line_pts, starting_points)
    
    ## We create the final fig ##
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Heuristic result')
    polygon = patches.Polygon(points, closed = True, fill = False)
    ax.add_artist(polygon)
    for ag in agents_allocation :

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
