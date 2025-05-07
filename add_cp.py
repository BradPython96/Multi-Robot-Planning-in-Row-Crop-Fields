import random
from math import dist
import numpy as np
from itertools import combinations

from field_and_rows import *
from Utils import Agent, Traveling, Charging, RATIO_ENERGY_OUTSIDE_FIELD

INF=np.inf

MODE = "Optimal Charging Allocator" # "test" "Compare Charging Allocators" "Optimal Charging Allocator"
NB_SIDE_POLYGON = 10
NB_ROBOTS = 3
NB_CHARGING_POINTS=10
NB_ROWS=50
ROWS_CROSS_INDEX=1 # TODO : Change to adapt to rows heterogeneity

MAX_CHARGE=15
SAFETY_MARGIN=0.05
SPEED=1
SPEED_ON_ROW=1
SPEED_ON_TRANSITION=1

INIT_TH=0.1
INCR_TH=1.01

TIME_TO_RECHARGE_PER_PERCENT=0.1 # Time to recharge a percent of battery
RATIO_ENERGY_ON_TRANSITION=1

PRINT_FIG=1

colors=['b','g','r','c','m','y']

def random_charging_points(nb_cp):
    i = 0
    charging_points=[]
    while i<nb_cp:
        x_rand=random.uniform(-1.5,6.5)
        y_rand=random.uniform(-1.5,6.5)
        if (x_rand<-1 or x_rand>6) and (y_rand<-1 or y_rand>6):
            charging_points.append([x_rand,y_rand])
            i+=1
    return charging_points

def add_charging_points_greedy(line_pts, agents: list[Agent], charging_points, dist_cp_matrix):

    for ag in agents:
        # rob_alloc : [id, block_id, entry_point, distance to entry_point, distance to cover block, total distance]
        ag.remove_actions()
        actions=[]
                
        # From init_pos to end of first row
            # We compute the distance from the end of first row to the closest cp

        ind_closest_cp=0
        ag.__repr__()
        if ag.rows==None or len(ag.rows)==0:
            return 1
        dist_closest_cp = dist_cp_matrix[ind_closest_cp][ag.rows[0].id][(ag.ind_entry_p+1)%2]
        for ind_cp in range(1,len(charging_points)):
            dist_cp=dist_cp_matrix[ind_cp][ag.rows[0].id][(ag.ind_entry_p+1)%2]
            if dist_cp<dist_closest_cp:
                ind_closest_cp=ind_cp
                dist_closest_cp=dist_cp

    
        if (RATIO_ENERGY_OUTSIDE_FIELD*(ag.distance_to_entry_point+dist_closest_cp)+ag.rows[0].energy_expended_to_cross)>ag.init_charge-SAFETY_MARGIN*ag.max_charge: # Not enough energy to go from init pos to end of first row and then charge


            ind_closest_cp_from_init_pos=0
            dist_closest_cp_from_init_pos = dist(charging_points[ind_closest_cp_from_init_pos], ag.starting_point)
            for ind_cp in range(1,len(charging_points)):          
                dist_cp=dist(charging_points[ind_cp], ag.starting_point)
                if dist_cp<dist_closest_cp_from_init_pos:
                    ind_closest_cp_from_init_pos=ind_cp
                    dist_closest_cp_from_init_pos=dist_cp

            if dist_closest_cp_from_init_pos>ag.current_charge:
                print("NO POSSIBLE SOLUTION")
                return 0
            else:
                sp_to_cp=Traveling(0,
                                   ag.starting_point,
                                   charging_points[ind_closest_cp_from_init_pos],
                                   dist_closest_cp_from_init_pos,
                                   dist_closest_cp_from_init_pos)
                actions.append(sp_to_cp)
                ag.compute_current_charge(sp_to_cp)

                charging=Charging(charging_points[ind_closest_cp_from_init_pos], ag.current_charge, ag.max_charge)
                actions.append(charging)
                ag.compute_current_charge(charging)

                cp_to_entry_p=Traveling(0,
                                        charging_points[ind_closest_cp_from_init_pos],
                                        ag.entry_point, 
                                        dist_cp_matrix[ind_closest_cp_from_init_pos][ag.rows[0].id][ag.ind_entry_p], 
                                        dist_cp_matrix[ind_closest_cp_from_init_pos][ag.rows[0].id][ag.ind_entry_p]) # Energy used = distance
                actions.append(cp_to_entry_p) 
                ag.compute_current_charge(cp_to_entry_p)

                next_row=Traveling(1,
                                   ag.entry_point,
                                   line_pts[ag.rows[0].id][(ag.ind_entry_p+1)%2],
                                   ag.rows[0].length,
                                   ag.rows[0].energy_expended_to_cross)
                
                actions.append(next_row)
                ag.compute_current_charge(next_row)
                last_ind=(ag.ind_entry_p+1)%2
                
        else: # Enough energy
            sp_to_entry_p=Traveling(0,
                                    ag.starting_point,
                                    ag.entry_point,
                                    ag.distance_to_entry_point,
                                    ag.distance_to_entry_point)
            actions.append(sp_to_entry_p)
            ag.compute_current_charge(sp_to_entry_p)

            next_row=Traveling(1,
                               ag.entry_point,
                               line_pts[ag.rows[0].id][(ag.ind_entry_p+1)%2],
                               ag.rows[0].length,
                               ag.rows[0].energy_expended_to_cross)
            actions.append(next_row)
            ag.compute_current_charge(next_row)
            last_ind=(ag.ind_entry_p+1)%2

            # For each row of the block, we check if we have enough energy to cross it and then go to cp

        for row in ag.rows[1:]:

            
            last_pos=actions[-1].end_pt

            ind_closest_next_cp=0
            dist_closest_next_cp = dist_cp_matrix[ind_closest_next_cp][row.id][(last_ind+1)%2]
            for ind_cp in range(1,len(charging_points)):
                dist_cp=dist_cp_matrix[ind_cp][row.id][(last_ind+1)%2]
                if dist_cp<dist_closest_next_cp:
                    ind_closest_next_cp=ind_cp
                    dist_closest_next_cp=dist_cp


            if (row.energy_expended_to_cross+dist_closest_next_cp*RATIO_ENERGY_OUTSIDE_FIELD)>ag.current_charge-SAFETY_MARGIN*ag.max_charge:

                to_cp=Traveling(0,
                                last_pos,
                                charging_points[ind_closest_cp],
                                dist_closest_cp,
                                dist_closest_cp)
                actions.append(to_cp)
                ag.compute_current_charge(to_cp)
                charging=Charging(charging_points[ind_closest_cp],
                                        ag.current_charge,
                                        ag.max_charge)
                actions.append(charging)
                ag.compute_current_charge(charging)

                from_cp=Traveling(0,
                                  charging_points[ind_closest_cp],
                                  line_pts[row.id][last_ind],
                                  dist_cp_matrix[ind_closest_cp][row.id][last_ind],
                                  dist_cp_matrix[ind_closest_cp][row.id][last_ind])
                actions.append(from_cp)
                ag.compute_current_charge(from_cp)

                next_row=Traveling(1,
                                   line_pts[row.id][last_ind],
                                   line_pts[row.id][(last_ind+1)%2],
                                   row.length,
                                   row.energy_expended_to_cross)
                actions.append(next_row)
                ag.compute_current_charge(next_row)

            else:

                transi_dist=dist(last_pos, line_pts[row.id][last_ind])
                transi=Traveling(2,
                                 last_pos,
                                 line_pts[row.id][last_ind],
                                 transi_dist,
                                 transi_dist)
                actions.append(transi)
                ag.compute_current_charge(transi)

                next_row=Traveling(1,
                                   line_pts[row.id][last_ind],
                                   line_pts[row.id][(last_ind+1)%2],
                                   row.length,
                                   row.energy_expended_to_cross)
                actions.append(next_row)
                ag.compute_current_charge(next_row)

            ind_closest_cp=ind_closest_next_cp
            dist_closest_cp=dist_closest_next_cp
            last_ind=(last_ind+1)%2

        last_pos=actions[-1].end_pt
        to_last_cp=Traveling(0,
                             last_pos,
                             charging_points[ind_closest_cp],
                             dist_closest_cp,
                             dist_closest_cp)
        actions.append(to_last_cp)
        ag.compute_current_charge(to_last_cp)
        ag.add_actions(actions)

    return 1


def add_charging_points(line_pts, agents, charging_points, dist_cp_matrix):

    max_dist_row_cp=0
    for cp in dist_cp_matrix:
        for row in cp:
            for dist_row_cp in row:
                if dist_row_cp>max_dist_row_cp:
                    max_dist_row_cp=dist_row_cp

    # Optimize using Max2Others method :
        # Compute the minimal number of charge needed by each robot to finish the mission and go back to init pos ((dist-init_charge)%(max_charge))

    for ag in agents:

        actions=[]
        # Compute minimal number of charging states needed for the mission
        ind_closest_final_cp=0
        dist_closest_final_cp = dist_cp_matrix[ind_closest_final_cp][ag.rows[-1].id][ag.ind_out_p]
        for ind_cp in range(1,len(charging_points)):
            dist_cp=dist_cp_matrix[ind_cp][ag.rows[-1].id][ag.ind_out_p]
            if dist_cp<dist_closest_final_cp:
                ind_closest_final_cp=ind_cp
                dist_closest_final_cp=dist_cp

        total_dist=ag.block_total_dist()+dist_closest_final_cp

        # print("Agent : "+str(ag.id))
        if total_dist<ag.init_charge-SAFETY_MARGIN*ag.max_charge:
            mini_nb_cp=0
        else:
            mini_nb_cp=1+(total_dist-ag.init_charge)//(ag.max_charge)

        actions=[]
        actions.append(Traveling(1,
                                 line_pts[ag.rows[0].id][ag.ind_entry_p],
                                 line_pts[ag.rows[0].id][(ag.ind_entry_p+1)%2],
                                 ag.rows[0].length,
                                 ag.rows[0].energy_expended_to_cross))
        last_ind= (ag.ind_entry_p+1)%2

        for row in ag.rows[1:]:
            actions.append(Traveling(1,
                                     line_pts[row.id][last_ind], 
                                     line_pts[row.id][(last_ind+1)%2], 
                                     row.length,
                                     row.energy_expended_to_cross))
            last_ind= (last_ind+1)%2

        ind_closest_cp_from_init_pos=0
        dist_closest_cp_from_init_pos = dist(charging_points[ind_closest_cp_from_init_pos], ag.starting_point)
        for ind_cp in range(1,len(charging_points)):          
            dist_cp=dist(charging_points[ind_cp], ag.starting_point)
            if dist_cp<dist_closest_cp_from_init_pos:
                ind_closest_cp_from_init_pos=ind_cp
                dist_closest_cp_from_init_pos=dist_cp

        th=INIT_TH
        heuristic_failed=True
        to_print=None
        while heuristic_failed :
            nb_cp=0
            actions_to_insert=[]
            heuristic_failed=False

            ag.current_charge=ag.init_charge

            discriminant = ag.current_charge+dist_closest_cp_from_init_pos

            if ag.current_charge-SAFETY_MARGIN*ag.max_charge<dist_closest_cp_from_init_pos*RATIO_ENERGY_OUTSIDE_FIELD:
                heuristic_failed=False
                to_print="No Possible Solution"

            elif discriminant<=th:
                
                travel_to_cp=Traveling(0,
                                       ag.starting_point,
                                       charging_points[ind_closest_cp_from_init_pos],
                                       dist_closest_cp_from_init_pos,
                                       dist_closest_cp_from_init_pos)
                
                actions_to_insert.append([0,travel_to_cp])
                ag.compute_current_charge(travel_to_cp)
                if ag.current_charge<0:
                    print("ERROR ON HEURSTIC, CHECK LINE 429")
                    heuristic_failed=False
                    break
                charging=Charging(charging_points[ind_closest_cp_from_init_pos], ag.current_charge, ag.max_charge)
                actions_to_insert.append([1,charging])
                ag.compute_current_charge(charging)
                travel_to_entry_p=Traveling(0,
                                            charging_points[ind_closest_cp_from_init_pos],
                                            ag.entry_point,
                                            dist_closest_cp_from_init_pos,
                                            dist_closest_cp_from_init_pos)
                actions_to_insert.append([2,travel_to_entry_p])
                
                ag.compute_current_charge(travel_to_entry_p)
                
                nb_cp+=1

            else:
                travel_to_entry_p=Traveling(0,
                                            ag.starting_point,ag.entry_point,
                                            ag.distance_to_entry_point,
                                            ag.distance_to_entry_point)
                actions_to_insert.append([0,travel_to_entry_p ])
                ag.compute_current_charge(travel_to_entry_p)
            ag.compute_current_charge(actions[0])
        
            if not heuristic_failed:
                
                for action_ind in range(len(actions)):

                    len_actions_to_insert=len(actions_to_insert)

                    ind_closest_cp=0
                    dist_closest_cp = dist(actions[action_ind].end_pt, charging_points[ind_closest_cp])
                    for ind_cp in range(1,len(charging_points)):
                        dist_cp = dist(actions[action_ind].end_pt, charging_points[ind_cp])
                        if dist_cp<dist_closest_cp:
                            ind_closest_cp=ind_cp
                            dist_closest_cp=dist_cp
                    
                    discriminant = ag.current_charge+dist_closest_cp
                    
                    # if rob_alloc[0]==0:
                    #     print(energy, dist_closest_cp, discriminant, th)

                    if ag.current_charge-SAFETY_MARGIN*ag.max_charge<dist_closest_cp*RATIO_ENERGY_OUTSIDE_FIELD:
                            heuristic_failed=True
                            break

                    elif discriminant<=th: # We go to charging point
                        travel_to_cp=Traveling(0,
                                               actions[action_ind].end_pt, charging_points[ind_closest_cp],
                                               dist_closest_cp,
                                               dist_closest_cp)
                        actions_to_insert.append([action_ind+1+len_actions_to_insert, travel_to_cp])

                        ag.compute_current_charge(travel_to_cp)

                        if ag.current_charge<0:
                            print("ERROR ON HEURSTIC, CHECK LINE 488")
                            heuristic_failed=False
                            break

                        cp=Charging(charging_points[ind_closest_cp], ag.current_charge, ag.max_charge)
                        actions_to_insert.append([action_ind+2+len_actions_to_insert,cp])
                        ag.compute_current_charge(cp)
                        nb_cp+=1

                        if action_ind!=len(actions)-1: # If mission not finished, we go from charging point to next row
                            dist_from_cp=dist(charging_points[ind_closest_cp], actions[action_ind+1].start_pt)
                            travel_from_cp=Traveling(0,
                                                     charging_points[ind_closest_cp],
                                                     actions[action_ind+1].start_pt,
                                                     dist_from_cp,
                                                     dist_from_cp)
                            actions_to_insert.append([action_ind+3+len_actions_to_insert,travel_from_cp])
                            ag.compute_current_charge(travel_from_cp)
                            ag.compute_current_charge(actions[action_ind+1])

                    else: # We go to next row
                        if action_ind!=len(actions)-1:

                            dist_transition = dist(actions[action_ind].end_pt, actions[action_ind+1].start_pt)
                            transition=Traveling(2,
                                                 actions[action_ind].end_pt, actions[action_ind+1].start_pt,
                                                 dist_transition,
                                                 dist_transition)
                            actions_to_insert.append([action_ind+1+len_actions_to_insert, transition])
                            ag.compute_current_charge(transition)
                            ag.compute_current_charge(actions[action_ind+1])

            if nb_cp<mini_nb_cp:
                heuristic_failed=True
            
            th=th*INCR_TH

            if th>(MAX_CHARGE+max_dist_row_cp):
                print(th,(MAX_CHARGE+max_dist_row_cp))
                print("No possible solution !!")
                return 0
            
            
        if to_print!=None:
            print("WARNING : HEURISTIC FAILED")
            print(to_print)
            return 0
        else:
            # print("HEURISTIC SUCCEEDED")
            for act in actions_to_insert:
                actions.insert(act[0], act[1])

        if type(actions[-1])!=type(Charging(None,None,None)):
            actions.append(Traveling(0,
                                     actions[-1].end_pt,
                                     charging_points[ind_closest_final_cp],
                                     dist_closest_final_cp,
                                     dist_closest_final_cp))
        
        ag.add_actions(actions)
        # Compute makespans by looking for each robot the nearest charging point(s) that makes it finish the mission and go back to initial position

    # Return the list of waypoints of each robot

    return 1


def find_best_cp(start_pt, end_pt, charging_points):
    nearest_cp = charging_points[0]
    nearest_back_and_force=dist(start_pt, nearest_cp)+dist(nearest_cp, end_pt)
    for cp in charging_points[1:]:
        back_and_force=dist(start_pt, cp)+dist(cp, end_pt)
        if back_and_force<nearest_back_and_force:
            nearest_back_and_force=back_and_force
            nearest_cp=cp

    return cp, nearest_back_and_force


def add_charging_points_optimal(agents: list[Agent], charging_points, dist_cp_matrix):

    res=[]
    for ag in agents:
        print("\n===Agent "+str(ag.id),end="===\n")
        

        # Compute minimal number of charging states needed for the mission
        ag.find_last_cp(charging_points, dist_cp_matrix)
        

        minimal_energy_needed=ag.compute_mission_minimal_energy_needed()
        if minimal_energy_needed<ag.init_charge-SAFETY_MARGIN*ag.max_charge:
            best_possibility=None
            best_cp_to_go=None
            best_makespan_increase=0
            print("No cp needed")
        else:
            mini_nb_cp=int(1+(minimal_energy_needed-(ag.init_charge-SAFETY_MARGIN*ag.max_charge))//((1-SAFETY_MARGIN)*ag.max_charge))

            ag.compute_best_cp_before_rows(charging_points)
            ag.create_rows_transition_best_cp(charging_points, dist_cp_matrix)
            # print(ag.rows_transition_best_cp)

            mission_done=False
            nb_cp=mini_nb_cp


            while nb_cp<=len(charging_points) and not mission_done :
                
                print("Rows : ", end=" ")
                print([r.id for r in ag.rows])
                possibilities = combinations([i for i in range(len(ag.rows))],nb_cp)

                best_makespan_increase = INF
                best_possibility=None
                

                for p in possibilities:

                    cp_to_go=[]
                    print("possibility : ", end=" ")
                    print(p)
                    if p[0]==0:
                        cp_before_row=True
                        if ag.init_charge-SAFETY_MARGIN*ag.max_charge<RATIO_ENERGY_OUTSIDE_FIELD*ag.distance_best_cp_before_rows[0]:
                                print("IMPOSSIBLE SOLUTION : NOT ENOUGHT INIT CHARGE FOR AGENT n°"+str(ag.id))
                                return 0
                    else:
                        cp_before_row=False

                    makespan_increase = 0
                    first = True
                    try_possibility=True
                    for ind_row_after_cp in p:
                        if first :
                            first=False
                            if cp_before_row:
                                energy_to_cp=ag.distance_best_cp_before_rows[0]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_from_cp=ag.distance_best_cp_before_rows[1]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_to_enty_point=ag.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD
                                makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_to_enty_point
                                if makespan_increase>=best_makespan_increase:
                                    try_possibility=False
                                    break
                                cp_to_go.append(([ag.best_cp_before_rows, energy_to_cp, energy_from_cp],ag.init_charge-energy_to_cp))
                            else:
                                cp=ag.rows_transition_best_cp[ind_row_after_cp-1]
                                energy_to_cp=cp[1]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_rows=sum([r.energy_expended_to_cross for r in ag.rows[:ind_row_after_cp]])
                                dist_transitions=0
                                for ind_row in range(1,ind_row_after_cp):
                                    dist_transitions+=dist(ag.rows[ind_row-1].end_pt, ag.rows[ind_row].start_pt)
                                energy_transitions=RATIO_ENERGY_ON_TRANSITION*dist_transitions
                                charge_needed= RATIO_ENERGY_OUTSIDE_FIELD*ag.distance_to_entry_point+energy_rows+energy_transitions+energy_to_cp # Energie à dépenser pour atteindre le PREMIER cp
                                
                                if ag.init_charge-SAFETY_MARGIN*ag.max_charge<charge_needed:
                                    try_possibility=False
                                    break
                                else:
                                    print("cp : ", end=" ")
                                    print(cp)
                                    energy_from_cp=cp[2]*RATIO_ENERGY_OUTSIDE_FIELD

                                    # print(ag.rows[ind_row_after_cp].end_pt, ag.rows[ind_row_after_cp+1].start_pt)
                                    energy_transi=dist(ag.rows[ind_row_after_cp-1].end_pt, ag.rows[ind_row_after_cp].start_pt)*RATIO_ENERGY_ON_TRANSITION
                                    makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_transi
                                    if makespan_increase>=best_makespan_increase:
                                        try_possibility=False
                                        break
                                    print("energy_to_cp, energy_from_cp, energy_transi :", end=" ")
                                    print(energy_to_cp,energy_from_cp,energy_transi)
                                    cp_to_go.append((cp,ag.init_charge-charge_needed))
                            last_row=ind_row_after_cp
                        
                        else:
                            cp=ag.rows_transition_best_cp[ind_row_after_cp-1]
                            print("cp : ", end="")
                            print(cp)
                            energy_to_cp=cp[1]*RATIO_ENERGY_OUTSIDE_FIELD
                            energy_rows=sum([r.energy_expended_to_cross for r in ag.rows[last_row:ind_row_after_cp]])
                            print("ag.rows[last_row].id, ag.rows[ind_row_after_cp].id :",end=" ")
                            print(ag.rows[last_row].id,ag.rows[ind_row_after_cp].id)

                            dist_transitions=0
                            for ind_row in range(last_row,ind_row_after_cp-1):
                                # print(dist(ag.rows[ind_row].end_pt, ag.rows[ind_row+1].start_pt))
                                dist_transitions+=dist(ag.rows[ind_row].end_pt, ag.rows[ind_row+1].start_pt)
                            energy_transitions=RATIO_ENERGY_ON_TRANSITION*dist_transitions

                            charge_needed = energy_from_cp+energy_rows+energy_transitions+energy_to_cp # Energie à dépenser pour atteindre le PROCHAIN cp

                            print("energy_to_cp, energy_from_cp, energy_rows, energy_transitions : ",end="")
                            print(energy_to_cp,energy_from_cp,energy_rows,energy_transitions)
                            print("charge_needed, ag.max_charge*(1-SAFETY_MARGIN)", end=" ")
                            print(charge_needed, ag.max_charge*(1-SAFETY_MARGIN))
                            if ag.max_charge*(1-SAFETY_MARGIN)< charge_needed :
                                try_possibility=False
                                break
                            else:

                                energy_from_cp=cp[2]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_transi=dist(ag.rows[ind_row_after_cp-1].end_pt, ag.rows[ind_row_after_cp].start_pt)*RATIO_ENERGY_ON_TRANSITION

                                makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_transi
                                if makespan_increase>=best_makespan_increase:
                                    try_possibility=False
                                    break
                                print("energy_from_cp, energy_to_cp, energy_transi : ", end=" ")
                                print(energy_to_cp,energy_from_cp,energy_transi)
                                cp_to_go.append((cp,ag.max_charge-charge_needed))
                            last_row=ind_row_after_cp

                    if try_possibility:
                        if makespan_increase<best_makespan_increase:
                            best_possibility=p
                            best_cp_to_go=cp_to_go
                            best_makespan_increase=makespan_increase
                            mission_done=True
                            print("New best makespan (for cp) : ", best_makespan_increase)


                nb_cp+=1
        
        if best_makespan_increase==INF:
            print("NO POSSIBLE SOLUTION FOR AGENT "+str(ag.id))
            return 0
        else:
            print("res added : ", end="")
            print([ag.id, best_possibility, best_cp_to_go, best_makespan_increase])
            res.append([ag.id, best_possibility, best_cp_to_go, best_makespan_increase])

            next_cp_ind=0
            if best_possibility!=None and best_possibility[next_cp_ind]==0:
                to_cp=Traveling(0, ag.starting_point, best_cp_to_go[next_cp_ind][0][0],best_cp_to_go[next_cp_ind][0][1],best_cp_to_go[next_cp_ind][0][1]*RATIO_ENERGY_OUTSIDE_FIELD)
                ag.add_actions(to_cp)
                cp=Charging(best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][1], ag.max_charge)
                ag.add_actions(cp)
                from_cp=Traveling(0, best_cp_to_go[next_cp_ind][0][0], ag.entry_point, best_cp_to_go[next_cp_ind][0][2],best_cp_to_go[next_cp_ind][0][2]*RATIO_ENERGY_OUTSIDE_FIELD)
                ag.add_actions(from_cp)
                next_cp_ind+=1
            else:
                to_entry_p=Traveling(0, ag.starting_point, ag.entry_point, ag.distance_to_entry_point, ag.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD)
                ag.add_actions(to_entry_p)

            first_row=Traveling(1, ag.rows[0].start_pt, ag.rows[0].end_pt, ag.rows[0].length, ag.rows[0].energy_expended_to_cross)
            ag.add_actions(first_row)
            r_ind=1
            for r in ag.rows[1:]:
                
                if best_possibility!=None and next_cp_ind<len(best_possibility) and best_possibility[next_cp_ind]==r_ind:
                    to_cp=Traveling(0, ag.actions[-1].end_pt,  best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][0][1],best_cp_to_go[next_cp_ind][0][1]*RATIO_ENERGY_OUTSIDE_FIELD)
                    ag.add_actions(to_cp)
                    cp=Charging(best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][0][1], ag.max_charge)
                    ag.add_actions(cp)
                    from_cp=Traveling(0, best_cp_to_go[next_cp_ind][0][0], r.start_pt, best_cp_to_go[next_cp_ind][0][2], best_cp_to_go[0][0][2]*RATIO_ENERGY_OUTSIDE_FIELD)
                    ag.add_actions(from_cp)
                    next_cp_ind+=1
                else:
                    dist_transi=dist(ag.actions[-1].end_pt, r.start_pt)
                    transi=Traveling(2, ag.actions[-1].end_pt, r.start_pt, dist_transi , dist_transi*RATIO_ENERGY_ON_TRANSITION)
                    ag.add_actions(transi)
                cross_row=Traveling(1, r.start_pt, r.end_pt, r.length, r.energy_expended_to_cross)
                ag.add_actions(cross_row)
                r_ind+=1

            to_final_cp=Traveling(0, ag.actions[-1].end_pt, ag.final_cp, ag.dist_to_final_cp, ag.dist_to_final_cp*RATIO_ENERGY_OUTSIDE_FIELD)
            ag.add_actions(to_final_cp)


    return res

def compute_dist_cp_matrix(charging_points, line_pts):

    dist_cp_matrix=[]
    for cp in charging_points:
        dist_cp=[]
        for row in line_pts:
            dist_cp+=[[dist(cp, row[0]), dist(cp, row[1])]]
        dist_cp_matrix.append(dist_cp)
    return dist_cp_matrix

def compute_dist_sp_matrix(starting_points, line_pts) :
        dist_matrix=[]
        for sp in starting_points:
            dist_sp=[]
            for row in line_pts:
                dist_sp+=[dist(sp, row[0]), dist(sp, row[1])]
            dist_matrix.append(dist_sp)