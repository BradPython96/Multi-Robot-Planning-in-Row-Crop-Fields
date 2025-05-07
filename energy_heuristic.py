import random
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from copy import deepcopy
from no_energy_heuristic import simple_rows_allocation
from add_cp import add_charging_points_greedy, random_charging_points, compute_dist_cp_matrix
from field_and_rows import  random_starting_points, to_convex_contour, rows_creator_nb_rows
from Utils import Row, Agent, Traveling, Charging, draw_semicircle, give_row, compute_list_of_agents_makespan, compute_makespan_with_cp_greedy_hungarian

colors=['b','g','r','c','y']


def Heuristic_cp(agents, line_pts, starting_points,charging_points):

    simple_blocks, robot_lenght_to_cover, lengths = simple_rows_allocation(line_pts, len(starting_points))
    dist_cp_matrix=compute_dist_cp_matrix(charging_points, line_pts)
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

    agents, ms=compute_makespan_with_cp_greedy_hungarian(block_list,agents,charging_points, dist_cp_matrix, SAFETY_MARGIN)

    ms=0
    init_worst_agent_ind=0
    first_working_agent=0
    for ag in agents :
        
        ag_length=ag.compute_makespan()
        if ag_length>ms:
            ms=ag_length
            init_worst_agent_ind=ag.get_id()
        if ag_length==0:
            agents.insert(0, agents.pop(ag.get_id()))
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
                res=add_charging_points_greedy(line_pts, agents, charging_points, dist_cp_matrix)
                makespan = compute_list_of_agents_makespan(agents)[1]
                if res==1:
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

NB_SIDE_POLYGON = 8
NB_ROBOTS = 5
NB_CHARGING_POINTS=10
NB_ROWS=50
ROWS_CROSS_INDEX=1 # TODO : Change to adapt to rows heterogeneity
MAX_CHARGE=50
SAFETY_MARGIN=0.05


if __name__ == "__main__":
    rd_seed=6
    points = to_convex_contour(NB_SIDE_POLYGON, rd_seed)
    line_pts=rows_creator_nb_rows(points, NB_ROWS)

    starting_points = random_starting_points(NB_ROBOTS, rd_seed)

    charging_points=random_charging_points(NB_CHARGING_POINTS)


    id=0
    agents_list=[]
    for sp in starting_points:
        init_charge=random.randint(int(MAX_CHARGE/3),MAX_CHARGE)
        ag=Agent(id, sp, init_charge, MAX_CHARGE, None, None, None)
        agents_list.append(ag)
        id+=1

    # Calcul des matrices de distance rows/starting_points ET rows/charging_points
    dist_cp_matrix=compute_dist_cp_matrix(charging_points, line_pts)
    agents, makespan = Heuristic_cp(agents_list, line_pts, starting_points, charging_points, )

    ## We create the final fig ##
    fig, ax = plt.subplots(1, 1)
    ax.set_title('Local Optimum')
    polygon = patches.Polygon(points, closed = True, fill = False)
    ax.add_artist(polygon)
    for ag in agents :

        color = colors[ag.id]
        ax.plot(ag.starting_point[0], ag.starting_point[1], marker="X", markersize=15, color=color)
        ax.annotate(str(ag.init_charge)+"/"+str(ag.max_charge),xy=(ag.starting_point[0], ag.starting_point[1]+0.5))
        ind=0
        for stage in ag.actions:
            if type(stage)==type(Traveling(None,None,None, None, None)):
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
                    # ax_o.add_artist(lines.Line2D([stage.start_pt[0], stage.end_pt[0]],[stage.start_pt[1], stage.end_pt[1]], color=color))
            elif type(stage)==type(Charging(None, None, None)):
                ind+=1
                # ax.annotate(str(int(stage.energy_on_arrival*100)/100)+"->"+str(stage.energy_on_leaving),xy=(stage.pos[0], stage.pos[1]+0.5))

    for cp in charging_points :
        ax.plot(cp[0], cp[1], markersize=12,marker="s", color="m")
    border_x_min = min(cp[0] for cp in charging_points)
    border_x_max = max(cp[0] for cp in charging_points)
    border_y_min = min(cp[1] for cp in charging_points)
    border_y_max = max(cp[1] for cp in charging_points)

    ax.set_xlim(border_x_min-0.2, border_x_max+0.2)
    ax.set_ylim(border_y_min-0.2, border_y_max+0.2)
    print("Makespan : ",makespan)
    plt.show()