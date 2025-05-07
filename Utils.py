from math import dist
import numpy as np
from itertools import combinations
from matplotlib.patches import Arc
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import networkx as nx
from Munkres_MinMax import MunkresMinMax


INF=np.inf
ROWS_CROSS_INDEX=1 # TODO : Change to adapt to rows heterogeneity

SPEED=1
SPEED_ON_ROW=1
SPEED_ON_TRANSITION=1

TIME_TO_RECHARGE_PER_PERCENT=0.1 # Time to recharge a percent of battery
RATIO_ENERGY_OUTSIDE_FIELD=1
RATIO_ENERGY_ON_TRANSITION=1

class LineSeg():
    
    def __init__(self, start_pt, end_pt):
        self.x, self.y = start_pt
        self.x2, self.y2 = end_pt
        if self.x != self.x2:
            self.m = (self.y2 - self.y) / (self.x2 - self.x)
            self.b = self.y - self.m*self.x
            self.minv = min(self.x, self.x2)
            self.maxv = max(self.x, self.x2)
        else:
            self.m = None
            self.b = self.x
            self.minv = min(self.y, self.y2)
            self.maxv = max(self.y, self.y2)
            
    def length(self):
        return np.linalg.norm([self.x2-self.x, self.y2-self.y])
    
    def intersect_w_line(self, m, b):
        if m == self.m:
            return (None, None)
        elif m == None:
            if self.minv <= b <= self.maxv:
                return (b, self.m*b + self.b)
            else:
                return (None, None)
        elif self.m == None:
            y = m*self.b + b
            if self.minv <= y <= self.maxv:
                return (self.b, y)
            else:
                return (None, None)
        else:
            x = (b - self.b) / (self.m - m)
            y = self.m*x + self.b
            if self.minv <= x <= self.maxv:
                return (x, y)
            else:
                return (None, None)    
            
    def intercept_range(self, m):
        if self.m == m:
            return (self.b, self.b)
        elif m == None:
            return sorted([self.x, self.x2])
        else:
            b = self.y - m*self.x
            b2 = self.y2 - m*self.x2
            return sorted([b, b2])
        

class Row():
    def __init__(self, id, extremity_points, length, energy_expended_to_cross):
        self.id = id
        self.extremity_points = extremity_points
        self.start_pt=None
        self.end_pt=None
        self.length = length
        
        self.cross_index=ROWS_CROSS_INDEX
        self.energy_expended_to_cross = energy_expended_to_cross

    def __repr__(self) -> str:
        typename = type(self).__name__
        return f'{typename}({self.id})'
    def get_id(self):
        return self.id
    def get_extremity_points(self):
        return self.extremity_points
    
class SimpleAgent():
    def __init__(self, id, starting_point, rows):
        self.id = id
        self.starting_point = starting_point

        self.speed=SPEED
        self.speed_on_row=SPEED_ON_ROW
        self.speed_on_transi=SPEED_ON_TRANSITION

        self.rows=rows
        self.entry_point=None
        self.distance_to_entry_point=None
        # self.ind_entry_p, self.ind_out_p=self.compute_rows_order()

        # self.actions = []
        # self.makespan=0

    def __repr__(self) -> str:
        typename = type(self).__name__
        return f'{typename}({self.id})'
    
    def get_starting_point(self):
        return self.starting_point
    def set_rows(self, rows):
        self.rows=rows
    def set_entry_point(self, entry_p, distance_to_entry_p):
        self.entry_point=entry_p
        self.distance_to_entry_point=distance_to_entry_p

class Agent():
    def __init__(self, id, starting_point, init_charge, max_charge, rows, entry_point, distance_to_entry_point):
        self.id = id
        self.starting_point = starting_point

        self.init_charge = init_charge
        self.max_charge = max_charge
        self.current_charge=init_charge
        self.speed=SPEED
        self.speed_on_row=SPEED_ON_ROW
        self.speed_on_transi=SPEED_ON_TRANSITION
        self.ind_entry_p=None
        self.ind_out_p=None
        self.rows=rows # Will be ordered
        self.entry_point=entry_point
        self.distance_to_entry_point=distance_to_entry_point
        if rows!=None:
            self.non_ordered_rows=sorted(copy(rows), key=lambda x: x.get_id())
            self.compute_rows_order()

        else:
            self.non_ordered_rows=None
        
        self.final_cp=None
        self.dist_to_final_cp=None

        self.best_cp_before_rows=None
        self.distance_best_cp_before_rows=None
        self.rows_transition_best_cp=[]

        self.actions = []
        self.makespan=0

    def __del__(self):
        if 0:
            print("Agent "+str(self.id)+" updated.")

    def __repr__(self) -> str:
        typename = type(self).__name__
        return f'{typename}({self.id})'
    def print_rows(self):
        for r in self.rows:
            print(r.__repr__())
    def print_non_ordered_rows(self):
        print(self.non_ordered_rows)
    def get_id(self):
        return self.id
    def get_starting_point(self):
        return self.starting_point
    def remove_actions(self):
        self.actions=[]
    def create_actions(self):
        if self.entry_point!=None:
            to_entry_p=Traveling(0, self.starting_point, self.entry_point, self.distance_to_entry_point, self.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD)
            self.add_actions(to_entry_p)
            if len(self.rows)>=1:
                first_row=Traveling(1, self.rows[0].start_pt, self.rows[0].end_pt, self.rows[0].length, self.rows[0].energy_expended_to_cross)
                self.add_actions(first_row)
            else:
                print(self.entry_point, self.rows)
            r_ind=1
            for r in self.rows[1:]:
                dist_transi=dist(self.actions[-1].end_pt, r.start_pt)
                transi=Traveling(2, self.actions[-1].end_pt, r.start_pt, dist_transi , dist_transi*RATIO_ENERGY_ON_TRANSITION)
                self.add_actions(transi)
                cross_row=Traveling(1, r.start_pt, r.end_pt, r.length, r.energy_expended_to_cross)
                self.add_actions(cross_row)
                r_ind+=1

    def compute_entry_point(self):
        self.rows=copy(self.non_ordered_rows)
        if len(self.rows)==0:
            return None
        
        e_1=[dist(self.starting_point, self.rows[0].get_extremity_points()[0]), self.rows[0].get_extremity_points()[0]]
        e_2=[dist(self.starting_point, self.rows[0].get_extremity_points()[1]), self.rows[0].get_extremity_points()[1]]
        e_3=[dist(self.starting_point, self.rows[-1].get_extremity_points()[0]), self.rows[-1].get_extremity_points()[0]]
        e_4=[dist(self.starting_point, self.rows[-1].get_extremity_points()[1]), self.rows[-1].get_extremity_points()[1]]
        e_min=min(e_1,e_2,e_3,e_4, key= lambda x:x[0])

        self.entry_point=e_min[1]
        self.distance_to_entry_point=e_min[0]

    def compute_rows_order(self):
        self.rows=copy(self.non_ordered_rows)
        if len(self.rows)==0:
            return None, None
        if self.entry_point==self.rows[0].extremity_points[0]:
            ind_entry_p=0
            ind_out_p=(len(self.rows)%2)
        elif self.entry_point==self.rows[0].extremity_points[1]:
            ind_entry_p=1
            ind_out_p=((1+len(self.rows)%2)%2)
        elif self.entry_point==self.rows[-1].extremity_points[0]:
            self.rows.reverse()
            ind_entry_p=0
            ind_out_p=(len(self.rows)%2)
        elif self.entry_point==self.rows[-1].extremity_points[1]:
            self.rows.reverse()
            ind_entry_p=1
            ind_out_p=((1+len(self.rows)%2)%2)

        ind_p=ind_entry_p
        for row in self.rows:
            row.start_pt=row.extremity_points[ind_p]
            row.end_pt=row.extremity_points[(ind_p+1)%2]
            ind_p=(ind_p+1)%2

        self.ind_entry_p=ind_entry_p
        self.ind_out_p=ind_out_p


    def find_last_cp(self, charging_points, dist_cp_matrix):
        ind_closest_final_cp=0
        dist_closest_final_cp = dist_cp_matrix[ind_closest_final_cp][self.rows[-1].id][self.ind_out_p]

        for ind_cp in range(1,len(charging_points)):
            dist_cp=dist_cp_matrix[ind_cp][self.rows[-1].id][self.ind_out_p]
            if dist_cp<dist_closest_final_cp:
                ind_closest_final_cp=ind_cp
                dist_closest_final_cp=dist_cp

        self.final_cp=charging_points[ind_closest_final_cp]
        self.dist_to_final_cp=dist_closest_final_cp

        return 1

    def compute_best_cp_before_rows(self, charging_points):
        ind_closest_cp_from_init_pos=0
        dist_to_best_cp=dist(charging_points[ind_closest_cp_from_init_pos], self.starting_point)
        dist_from_best_cp=dist(charging_points[ind_closest_cp_from_init_pos], self.entry_point)

        dist_closest_cp_from_init_pos = dist_to_best_cp+dist_from_best_cp
        for ind_cp in range(1,len(charging_points)):
            dist_to_cp=dist(charging_points[ind_cp], self.starting_point)        
            dist_from_cp=dist(charging_points[ind_cp], self.entry_point)  
            dist_cp=dist_to_cp+dist_from_cp
            if dist_cp<dist_closest_cp_from_init_pos:
                ind_closest_cp_from_init_pos=ind_cp
                dist_to_best_cp=dist_to_cp
                dist_from_best_cp=dist_from_cp
                dist_closest_cp_from_init_pos=dist_cp

        self.best_cp_before_rows=charging_points[ind_closest_cp_from_init_pos]
        self.distance_best_cp_before_rows=[dist_to_best_cp,dist_from_best_cp]

        return 1
    
    def compute_best_cp_before_rows_with_border(self, nav_graph, charging_points):
        ind_closest_cp_from_init_pos=0
        dist_to_best_cp=dist(charging_points[ind_closest_cp_from_init_pos], self.starting_point)
        dist_from_best_cp=compute_shortest_path_length(nav_graph, np.array(charging_points[ind_closest_cp_from_init_pos]), np.array(self.entry_point))

        dist_closest_cp_from_init_pos = dist_to_best_cp+dist_from_best_cp
        for ind_cp in range(1,len(charging_points)):
            dist_to_cp=dist(charging_points[ind_cp], self.starting_point)        
            dist_from_cp=compute_shortest_path_length(nav_graph, np.array(charging_points[ind_cp]), np.array(self.entry_point))  
            dist_cp=dist_to_cp+dist_from_cp
            if dist_cp<dist_closest_cp_from_init_pos:
                ind_closest_cp_from_init_pos=ind_cp
                dist_to_best_cp=dist_to_cp
                dist_from_best_cp=dist_from_cp
                dist_closest_cp_from_init_pos=dist_cp

        self.best_cp_before_rows=charging_points[ind_closest_cp_from_init_pos]
        self.distance_best_cp_before_rows=[dist_to_best_cp,dist_from_best_cp]

        return 1

    def create_rows_transition_best_cp(self, charging_points, dist_cp_matrix):
        point_id=(self.ind_entry_p+1)%2
        for ind_row in range(1,len(self.rows)):
            ind_best_cp=0
            best_dist_to_cp= dist_cp_matrix[ind_best_cp][self.rows[ind_row-1].id][point_id]
            best_dist_from_cp=dist_cp_matrix[ind_best_cp][self.rows[ind_row].id][point_id]
            dist_best_cp =best_dist_to_cp+best_dist_from_cp

            for ind_cp in range(1,len(charging_points)):
                dist_to_cp=dist_cp_matrix[ind_cp][self.rows[ind_row-1].id][point_id]
                dist_from_cp=dist_cp_matrix[ind_cp][self.rows[ind_row].id][point_id]
                dist_cp = dist_to_cp+dist_from_cp
                if dist_cp<dist_best_cp:
                    ind_best_cp=ind_cp
                    best_dist_to_cp=dist_to_cp
                    best_dist_from_cp=dist_from_cp
                    dist_best_cp=dist_cp

            self.rows_transition_best_cp.append([charging_points[ind_best_cp], best_dist_to_cp, best_dist_from_cp])
            point_id=(point_id+1)%2

        return 1

    def compute_mission_minimal_energy_needed(self):
        res= self.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD
        first=True
        ind_pt=self.ind_entry_p
        for ind_row in range(len(self.rows)):
            if first:
                first=False
            else:
                ind_pt=(ind_pt+1)%2
                res+=self.rows[ind_row].energy_expended_to_cross
                res+=dist(self.rows[ind_row-1].extremity_points[ind_pt], self.rows[ind_row].extremity_points[ind_pt])*RATIO_ENERGY_ON_TRANSITION
        res+=self.dist_to_final_cp*RATIO_ENERGY_OUTSIDE_FIELD
        return res

    def compute_current_charge(self, actions):
        if type(actions)==type([]):
            for act in actions:
                if type(act)==type(Traveling(None,None,None, None, None)):
                    self.current_charge-=act.energy_expended

                elif type(act)==type(Charging(None, None, None)):
                    self.current_charge=act.energy_on_leaving


        elif type(actions)==type(Traveling(None,None,None, None, None)):
            self.current_charge-=actions.energy_expended
            # last_action=[actions.type_of_travel, actions.start_pt, actions.end_pt, actions.energy_expended]

        elif type(actions)==type(Charging(None, None, None)):
            self.current_charge=actions.energy_on_leaving
            # last_action=[actions.pos, actions.energy_on_arrival, actions.energy_on_leaving]


    def compute_makespan(self): # TODO Take into account the type of travel on makespan compute
        row_ind=0
        self.makespan=0
        if len(self.actions)==0:
            self.makespan=0
            return self.makespan
        for action in self.actions[:-1]:
            if type(action)==type(Traveling(None,None,None, None, None)):
                if action.type_of_travel==0: # Travel out of field
                    self.makespan+=action.distance/self.speed

                elif action.type_of_travel==1: # Cross row
                    self.makespan+=action.distance/self.speed_on_row*self.rows[row_ind].cross_index
                    row_ind+=1

                elif  action.type_of_travel==2: # Transition between 2 rows
                    self.makespan+=action.distance/self.speed_on_transi

            elif type(action)==type(Charging(None, None, None)):
                self.makespan+=(action.energy_on_leaving-action.energy_on_arrival)*TIME_TO_RECHARGE_PER_PERCENT

        if type(self.actions[-1])==type(Traveling(None,None,None, None, None)): # Travel to last cp
            self.makespan+=action.distance/self.speed
        return self.makespan

    def add_actions(self, actions):
        if type(actions)==type([]):
            self.actions+=actions
        else:
            self.actions.append(actions)

    def compute_optimal_cp(self, charging_points, dist_cp_matrix, SAFETY_MARGIN):
        self.remove_actions()
        self.find_last_cp(charging_points, dist_cp_matrix)
        
        minimal_energy_needed=self.compute_mission_minimal_energy_needed()
        if minimal_energy_needed<self.init_charge-SAFETY_MARGIN*self.max_charge:
            best_possibility=None
            best_cp_to_go=None
            best_makespan_increase=0
        else:
            mini_nb_cp=int(1+(minimal_energy_needed-(self.init_charge-SAFETY_MARGIN*self.max_charge))//((1-SAFETY_MARGIN)*self.max_charge))
            if mini_nb_cp<1:
                print(mini_nb_cp)
            self.compute_best_cp_before_rows(charging_points)
            self.create_rows_transition_best_cp(charging_points, dist_cp_matrix)

            mission_done=False
            nb_cp=mini_nb_cp


            while nb_cp<=len(charging_points) and not mission_done :
                
                possibilities = combinations([i for i in range(len(self.rows))],nb_cp)

                best_makespan_increase = INF
                best_possibility=None
                
                for p in possibilities:

                    cp_to_go=[]
                    if p[0]==0:
                        cp_before_row=True
                        if self.init_charge-SAFETY_MARGIN*self.max_charge<RATIO_ENERGY_OUTSIDE_FIELD*self.distance_best_cp_before_rows[0]:
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
                                energy_to_cp=self.distance_best_cp_before_rows[0]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_from_cp=self.distance_best_cp_before_rows[1]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_to_enty_point=self.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD
                                makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_to_enty_point
                                if makespan_increase>=best_makespan_increase:
                                    try_possibility=False
                                    break
                                cp_to_go.append(([self.best_cp_before_rows, energy_to_cp, energy_from_cp],self.init_charge-energy_to_cp))
                            else:
                                cp=self.rows_transition_best_cp[ind_row_after_cp-1]
                                energy_to_cp=cp[1]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_rows=sum([r.energy_expended_to_cross for r in self.rows[:ind_row_after_cp]])
                                dist_transitions=0
                                for ind_row in range(1,ind_row_after_cp):
                                    dist_transitions+=dist(self.rows[ind_row-1].end_pt, self.rows[ind_row].start_pt)
                                energy_transitions=RATIO_ENERGY_ON_TRANSITION*dist_transitions
                                charge_needed= RATIO_ENERGY_OUTSIDE_FIELD*self.distance_to_entry_point+energy_rows+energy_transitions+energy_to_cp # Energie à dépenser pour atteindre le PREMIER cp
                                
                                if self.init_charge-SAFETY_MARGIN*self.max_charge<charge_needed:
                                    try_possibility=False
                                    break
                                else:
                                    energy_from_cp=cp[2]*RATIO_ENERGY_OUTSIDE_FIELD

                                    energy_transi=dist(self.rows[ind_row_after_cp-1].end_pt, self.rows[ind_row_after_cp].start_pt)*RATIO_ENERGY_ON_TRANSITION
                                    makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_transi
                                    if makespan_increase>=best_makespan_increase:
                                        try_possibility=False
                                        break
                                    cp_to_go.append((cp,self.init_charge-charge_needed))
                            last_row=ind_row_after_cp
                        
                        else:
                            cp=self.rows_transition_best_cp[ind_row_after_cp-1]
                            energy_to_cp=cp[1]*RATIO_ENERGY_OUTSIDE_FIELD
                            energy_rows=sum([r.energy_expended_to_cross for r in self.rows[last_row:ind_row_after_cp]])

                            dist_transitions=0
                            for ind_row in range(last_row,ind_row_after_cp-1):
                                dist_transitions+=dist(self.rows[ind_row].end_pt, self.rows[ind_row+1].start_pt)
                            energy_transitions=RATIO_ENERGY_ON_TRANSITION*dist_transitions

                            charge_needed = energy_from_cp+energy_rows+energy_transitions+energy_to_cp # Energie à dépenser pour atteindre le PROCHAIN cp
                            if self.max_charge*(1-SAFETY_MARGIN)< charge_needed :
                                try_possibility=False
                                break
                            else:

                                energy_from_cp=cp[2]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_transi=dist(self.rows[ind_row_after_cp-1].end_pt, self.rows[ind_row_after_cp].start_pt)*RATIO_ENERGY_ON_TRANSITION

                                makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_transi
                                if makespan_increase>=best_makespan_increase:
                                    try_possibility=False
                                    break
                                cp_to_go.append((cp,self.max_charge-charge_needed))
                            last_row=ind_row_after_cp

                    if try_possibility:
                        if makespan_increase<best_makespan_increase:
                            best_possibility=p
                            best_cp_to_go=cp_to_go
                            best_makespan_increase=makespan_increase
                            mission_done=True
                nb_cp+=1
        if best_makespan_increase==INF:
            return 0
        else:

            next_cp_ind=0
            if best_possibility!=None and best_possibility[next_cp_ind]==0:
                to_cp=Traveling(0, self.starting_point, best_cp_to_go[next_cp_ind][0][0],best_cp_to_go[next_cp_ind][0][1],best_cp_to_go[next_cp_ind][0][1]*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(to_cp)
                cp=Charging(best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][1], self.max_charge)
                self.add_actions(cp)
                from_cp=Traveling(0, best_cp_to_go[next_cp_ind][0][0], self.entry_point, best_cp_to_go[next_cp_ind][0][2],best_cp_to_go[next_cp_ind][0][2]*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(from_cp)
                next_cp_ind+=1
            else:
                to_entry_p=Traveling(0, self.starting_point, self.entry_point, self.distance_to_entry_point, self.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(to_entry_p)

            first_row=Traveling(1, self.rows[0].start_pt, self.rows[0].end_pt, self.rows[0].length, self.rows[0].energy_expended_to_cross)
            self.add_actions(first_row)
            r_ind=1
            for r in self.rows[1:]:
                
                if best_possibility!=None and next_cp_ind<len(best_possibility) and best_possibility[next_cp_ind]==r_ind:
                    to_cp=Traveling(0, self.actions[-1].end_pt,  best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][0][1],best_cp_to_go[next_cp_ind][0][1]*RATIO_ENERGY_OUTSIDE_FIELD)
                    self.add_actions(to_cp)
                    cp=Charging(best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][0][1], self.max_charge)
                    self.add_actions(cp)
                    from_cp=Traveling(0, best_cp_to_go[next_cp_ind][0][0], r.start_pt, best_cp_to_go[next_cp_ind][0][2], best_cp_to_go[0][0][2]*RATIO_ENERGY_OUTSIDE_FIELD)
                    self.add_actions(from_cp)
                    next_cp_ind+=1
                else:
                    dist_transi=dist(self.actions[-1].end_pt, r.start_pt)
                    transi=Traveling(2, self.actions[-1].end_pt, r.start_pt, dist_transi , dist_transi*RATIO_ENERGY_ON_TRANSITION)
                    self.add_actions(transi)
                cross_row=Traveling(1, r.start_pt, r.end_pt, r.length, r.energy_expended_to_cross)
                self.add_actions(cross_row)
                r_ind+=1

            to_final_cp=Traveling(0, self.actions[-1].end_pt, self.final_cp, self.dist_to_final_cp, self.dist_to_final_cp*RATIO_ENERGY_OUTSIDE_FIELD)
            self.add_actions(to_final_cp)
        return 1

    def compute_optimal_cp_and_border(self, nav_graph, charging_points, dist_cp_matrix, SAFETY_MARGIN):
        self.remove_actions()
        self.find_last_cp(charging_points, dist_cp_matrix)
        
        minimal_energy_needed=self.compute_mission_minimal_energy_needed()
        if minimal_energy_needed<self.init_charge-SAFETY_MARGIN*self.max_charge:
            best_possibility=None
            best_cp_to_go=None
            best_makespan_increase=0
        else:
            mini_nb_cp=int(1+(minimal_energy_needed-(self.init_charge-SAFETY_MARGIN*self.max_charge))//((1-SAFETY_MARGIN)*self.max_charge))
            if mini_nb_cp<1:
                print(mini_nb_cp)
            self.compute_best_cp_before_rows_with_border(nav_graph, charging_points)
            self.create_rows_transition_best_cp(charging_points, dist_cp_matrix)

            mission_done=False
            nb_cp=mini_nb_cp


            while nb_cp<=len(charging_points) and not mission_done :
                
                possibilities = combinations([i for i in range(len(self.rows))],nb_cp)

                best_makespan_increase = INF
                best_possibility=None
                
                for p in possibilities:

                    cp_to_go=[]
                    if p[0]==0:
                        cp_before_row=True
                        if self.init_charge-SAFETY_MARGIN*self.max_charge<RATIO_ENERGY_OUTSIDE_FIELD*self.distance_best_cp_before_rows[0]:
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
                                energy_to_cp=self.distance_best_cp_before_rows[0]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_from_cp=self.distance_best_cp_before_rows[1]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_to_enty_point=self.distance_to_entry_point*RATIO_ENERGY_OUTSIDE_FIELD
                                makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_to_enty_point
                                if makespan_increase>=best_makespan_increase:
                                    try_possibility=False
                                    break
                                cp_to_go.append(([self.best_cp_before_rows, energy_to_cp, energy_from_cp],self.init_charge-energy_to_cp))
                            else:
                                cp=self.rows_transition_best_cp[ind_row_after_cp-1]
                                energy_to_cp=cp[1]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_rows=sum([r.energy_expended_to_cross for r in self.rows[:ind_row_after_cp]])
                                dist_transitions=0
                                for ind_row in range(1,ind_row_after_cp):
                                    dist_transitions+=dist(self.rows[ind_row-1].end_pt, self.rows[ind_row].start_pt)
                                energy_transitions=RATIO_ENERGY_ON_TRANSITION*dist_transitions
                                charge_needed= RATIO_ENERGY_OUTSIDE_FIELD*self.distance_to_entry_point+energy_rows+energy_transitions+energy_to_cp # Energie à dépenser pour atteindre le PREMIER cp
                                
                                if self.init_charge-SAFETY_MARGIN*self.max_charge<charge_needed:
                                    try_possibility=False
                                    break
                                else:
                                    energy_from_cp=cp[2]*RATIO_ENERGY_OUTSIDE_FIELD

                                    energy_transi=dist(self.rows[ind_row_after_cp-1].end_pt, self.rows[ind_row_after_cp].start_pt)*RATIO_ENERGY_ON_TRANSITION
                                    makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_transi
                                    if makespan_increase>=best_makespan_increase:
                                        try_possibility=False
                                        break
                                    cp_to_go.append((cp,self.init_charge-charge_needed))
                            last_row=ind_row_after_cp
                        
                        else:
                            cp=self.rows_transition_best_cp[ind_row_after_cp-1]
                            energy_to_cp=cp[1]*RATIO_ENERGY_OUTSIDE_FIELD
                            energy_rows=sum([r.energy_expended_to_cross for r in self.rows[last_row:ind_row_after_cp]])

                            dist_transitions=0
                            for ind_row in range(last_row,ind_row_after_cp-1):
                                dist_transitions+=dist(self.rows[ind_row].end_pt, self.rows[ind_row+1].start_pt)
                            energy_transitions=RATIO_ENERGY_ON_TRANSITION*dist_transitions

                            charge_needed = energy_from_cp+energy_rows+energy_transitions+energy_to_cp # Energie à dépenser pour atteindre le PROCHAIN cp
                            if self.max_charge*(1-SAFETY_MARGIN)< charge_needed :
                                try_possibility=False
                                break
                            else:

                                energy_from_cp=cp[2]*RATIO_ENERGY_OUTSIDE_FIELD
                                energy_transi=dist(self.rows[ind_row_after_cp-1].end_pt, self.rows[ind_row_after_cp].start_pt)*RATIO_ENERGY_ON_TRANSITION

                                makespan_increase=makespan_increase+energy_to_cp+energy_from_cp-energy_transi
                                if makespan_increase>=best_makespan_increase:
                                    try_possibility=False
                                    break
                                cp_to_go.append((cp,self.max_charge-charge_needed))
                            last_row=ind_row_after_cp

                    if try_possibility:
                        if makespan_increase<best_makespan_increase:
                            best_possibility=p
                            best_cp_to_go=cp_to_go
                            best_makespan_increase=makespan_increase
                            mission_done=True
                nb_cp+=1

        if best_makespan_increase==INF:
            return 0
        else:

            next_cp_ind=0
            if best_possibility!=None and best_possibility[next_cp_ind]==0:

                distance=dist(self.starting_point, best_cp_to_go[next_cp_ind][0][0])
                to_cp=Traveling(0, self.starting_point, best_cp_to_go[next_cp_ind][0][0],distance, distance*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(to_cp)
                    
                cp=Charging(best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][1], self.max_charge)
                self.add_actions(cp)
                from_cp_waypoints=find_shortest_path(nav_graph, np.array(best_cp_to_go[next_cp_ind][0][0]), np.array(self.entry_point))
                for ind_wp in range(1,len(from_cp_waypoints)):
                    distance=dist(from_cp_waypoints[ind_wp-1], from_cp_waypoints[ind_wp])
                    from_cp=Traveling(0, from_cp_waypoints[ind_wp-1], from_cp_waypoints[ind_wp], distance, distance*RATIO_ENERGY_OUTSIDE_FIELD)
                    self.add_actions(from_cp)
                next_cp_ind+=1
            else:
                to_entry_p_waypoints=find_shortest_path(nav_graph, np.array(self.starting_point), np.array(self.entry_point))
                for ind_wp in range(1,len(to_entry_p_waypoints)):
                    distance=dist(to_entry_p_waypoints[ind_wp-1], to_entry_p_waypoints[ind_wp])
                    to_entry_p=Traveling(0,to_entry_p_waypoints[ind_wp-1], to_entry_p_waypoints[ind_wp], distance, distance*RATIO_ENERGY_OUTSIDE_FIELD)
                    self.add_actions(to_entry_p)


            first_row=Traveling(1, self.rows[0].start_pt, self.rows[0].end_pt, self.rows[0].length, self.rows[0].energy_expended_to_cross)
            self.add_actions(first_row)
            r_ind=1
            for r in self.rows[1:]:
                
                if best_possibility!=None and next_cp_ind<len(best_possibility) and best_possibility[next_cp_ind]==r_ind:

                    to_cp_waypoints=find_shortest_path(nav_graph, np.array(self.actions[-1].end_pt), np.array(best_cp_to_go[next_cp_ind][0][0]))
                    for wp_ind in range(1, len(to_cp_waypoints)):
                        distance=dist(to_cp_waypoints[wp_ind-1], to_cp_waypoints[wp_ind])
                        to_cp=Traveling(0, to_cp_waypoints[wp_ind-1], to_cp_waypoints[wp_ind], distance, distance*RATIO_ENERGY_OUTSIDE_FIELD)
                        self.add_actions(to_cp)

                    cp=Charging(best_cp_to_go[next_cp_ind][0][0], best_cp_to_go[next_cp_ind][0][1], self.max_charge)
                    self.add_actions(cp)
                    from_cp_waypoints=find_shortest_path(nav_graph, np.array(best_cp_to_go[next_cp_ind][0][0]), np.array(r.start_pt))
                    for wp_ind in range(1, len(from_cp_waypoints)):
                        distance=dist(from_cp_waypoints[wp_ind-1], from_cp_waypoints[wp_ind])
                        from_cp=Traveling(0,from_cp_waypoints[wp_ind-1], from_cp_waypoints[wp_ind], distance, distance*RATIO_ENERGY_OUTSIDE_FIELD)
                        self.add_actions(from_cp)
                    next_cp_ind+=1
                else:
                    dist_transi=dist(self.actions[-1].end_pt, r.start_pt)
                    transi=Traveling(2, self.actions[-1].end_pt, r.start_pt, dist_transi , dist_transi*RATIO_ENERGY_ON_TRANSITION)
                    self.add_actions(transi)
                cross_row=Traveling(1, r.start_pt, r.end_pt, r.length, r.energy_expended_to_cross)
                self.add_actions(cross_row)
                r_ind+=1


            to_final_cp_waypoints=find_shortest_path(nav_graph, self.actions[-1].end_pt, self.final_cp)
            for wp_ind in range(1, len(to_final_cp_waypoints)):
                distance=dist(to_final_cp_waypoints[wp_ind-1], to_final_cp_waypoints[wp_ind])
                to_final_cp=Traveling(0,to_final_cp_waypoints[wp_ind-1], to_final_cp_waypoints[wp_ind], distance, distance*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(to_final_cp)
            next_cp_ind+=1
            
        return 1

    def compute_greedy_cp(self, charging_points, dist_cp_matrix, SAFETY_MARGIN):
        self.remove_actions()
                
        # From init_pos to end of first row
            # We compute the distance from the end of first row to the closest cp

        ind_closest_cp=0
        self.__repr__()
        if self.rows==None or len(self.rows)==0:
            return 1
        dist_closest_cp = dist_cp_matrix[ind_closest_cp][self.rows[0].id][(self.ind_entry_p+1)%2]
        for ind_cp in range(1,len(charging_points)):
            dist_cp=dist_cp_matrix[ind_cp][self.rows[0].id][(self.ind_entry_p+1)%2]
            if dist_cp<dist_closest_cp:
                ind_closest_cp=ind_cp
                dist_closest_cp=dist_cp

    
        if (RATIO_ENERGY_OUTSIDE_FIELD*(self.distance_to_entry_point+dist_closest_cp)+self.rows[0].energy_expended_to_cross)>self.init_charge-SAFETY_MARGIN*self.max_charge: # Not enough energy to go from init pos to end of first row and then charge


            ind_closest_cp_from_init_pos=0
            dist_closest_cp_from_init_pos = dist(charging_points[ind_closest_cp_from_init_pos], self.starting_point)
            for ind_cp in range(1,len(charging_points)):          
                dist_cp=dist(charging_points[ind_cp], self.starting_point)
                if dist_cp<dist_closest_cp_from_init_pos:
                    ind_closest_cp_from_init_pos=ind_cp
                    dist_closest_cp_from_init_pos=dist_cp

            if dist_closest_cp_from_init_pos>self.current_charge:
                print("NO POSSIBLE SOLUTION")
                return 0
            else:
                sp_to_cp=Traveling(0,
                                   self.starting_point,
                                   charging_points[ind_closest_cp_from_init_pos],
                                   dist_closest_cp_from_init_pos,
                                   dist_closest_cp_from_init_pos)
                self.add_actions(sp_to_cp)
                self.compute_current_charge(sp_to_cp)

                charging=Charging(charging_points[ind_closest_cp_from_init_pos], self.current_charge, self.max_charge)
                self.add_actions(charging)
                self.compute_current_charge(charging)

                cp_to_entry_p=Traveling(0,
                                        charging_points[ind_closest_cp_from_init_pos],
                                        self.entry_point, 
                                        dist_cp_matrix[ind_closest_cp_from_init_pos][self.rows[0].id][self.ind_entry_p], 
                                        dist_cp_matrix[ind_closest_cp_from_init_pos][self.rows[0].id][self.ind_entry_p]) # Energy used = distance
                self.add_actions(cp_to_entry_p) 
                self.compute_current_charge(cp_to_entry_p)

                next_row=Traveling(1,
                                   self.rows[0].start_pt,
                                   self.rows[0].end_pt,
                                   self.rows[0].length,
                                   self.rows[0].energy_expended_to_cross)
                
                self.add_actions(next_row)
                self.compute_current_charge(next_row)
                last_ind=(self.ind_entry_p+1)%2
                
        else: # Enough energy
            sp_to_entry_p=Traveling(0,
                                    self.starting_point,
                                    self.entry_point,
                                    self.distance_to_entry_point,
                                    self.distance_to_entry_point)
            self.add_actions(sp_to_entry_p)
            self.compute_current_charge(sp_to_entry_p)

            next_row=Traveling(1,
                               self.rows[0].start_pt,
                               self.rows[0].end_pt,
                               self.rows[0].length,
                               self.rows[0].energy_expended_to_cross)
            self.add_actions(next_row)
            self.compute_current_charge(next_row)
            last_ind=(self.ind_entry_p+1)%2

            # For each row of the block, we check if we have enough energy to cross it and then go to cp

        for row in self.rows[1:]:

            
            last_pos=self.actions[-1].end_pt

            ind_closest_next_cp=0
            dist_closest_next_cp = dist_cp_matrix[ind_closest_next_cp][row.id][(last_ind+1)%2]
            for ind_cp in range(1,len(charging_points)):
                dist_cp=dist_cp_matrix[ind_cp][row.id][(last_ind+1)%2]
                if dist_cp<dist_closest_next_cp:
                    ind_closest_next_cp=ind_cp
                    dist_closest_next_cp=dist_cp


            if (row.energy_expended_to_cross+dist_closest_next_cp*RATIO_ENERGY_OUTSIDE_FIELD)>self.current_charge-SAFETY_MARGIN*self.max_charge:

                to_cp=Traveling(0,
                                last_pos,
                                charging_points[ind_closest_cp],
                                dist_closest_cp,
                                dist_closest_cp*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(to_cp)
                self.compute_current_charge(to_cp)
                charging=Charging(charging_points[ind_closest_cp],
                                        self.current_charge,
                                        self.max_charge)
                self.add_actions(charging)
                self.compute_current_charge(charging)

                from_cp=Traveling(0,
                                  charging_points[ind_closest_cp],
                                  row.start_pt,
                                  dist_cp_matrix[ind_closest_cp][row.id][last_ind],
                                  dist_cp_matrix[ind_closest_cp][row.id][last_ind]*RATIO_ENERGY_OUTSIDE_FIELD)
                self.add_actions(from_cp)
                self.compute_current_charge(from_cp)

                next_row=Traveling(1,
                                   row.start_pt,
                                   row.end_pt,
                                   row.length,
                                   row.energy_expended_to_cross)
                self.add_actions(next_row)
                self.compute_current_charge(next_row)

            else:
                dist_transi=dist(self.actions[-1].end_pt, row.start_pt)
                transi=Traveling(2, self.actions[-1].end_pt, row.start_pt, dist_transi , dist_transi*RATIO_ENERGY_ON_TRANSITION)

                self.add_actions(transi)
                self.compute_current_charge(transi)

                next_row=Traveling(1,
                                   row.start_pt,
                                   row.end_pt,
                                   row.length,
                                   row.energy_expended_to_cross)
                self.add_actions(next_row)
                self.compute_current_charge(next_row)

            ind_closest_cp=ind_closest_next_cp
            dist_closest_cp=dist_closest_next_cp
            last_ind=(last_ind+1)%2

        last_pos=self.actions[-1].end_pt
        to_last_cp=Traveling(0,
                             last_pos,
                             charging_points[ind_closest_cp],
                             dist_closest_cp,
                             dist_closest_cp)
        self.add_actions(to_last_cp)
        self.compute_current_charge(to_last_cp)
        return 1

class Traveling():
    def __init__(self, type_of_travel, start_pt, end_pt,distance, energy_expended):
        self.type_of_travel = type_of_travel # 0 : out of field, 1 : cross row, 2 : transition between two rows
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.distance = distance
        self.energy_expended=energy_expended

class Charging():
    def __init__(self, pos, energy_on_arrival, energy_on_leaving):
        self.pos = pos
        self.energy_on_arrival = energy_on_arrival
        self.energy_on_leaving = energy_on_leaving



def draw_semicircle(x1, y1, x2, y2, color='black', lw=1, ax=None):
    '''
    draw a semicircle between the points x1,y1 and x2,y2
    the semicircle is drawn to the left of the segment
    '''
    ax = ax or plt.gca()
    # ax. Scatter([x1, x2], [y1, y2], s=100, c=color)
    startangle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    diameter = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Euclidian distance
    ax.add_patch(Arc(((x1 + x2) / 2, (y1 + y2) / 2), diameter, diameter, theta1=startangle, theta2=startangle + 180,
                     edgecolor=color, facecolor='none', lw=lw, zorder=0))
    
def create_nav_graphs(inter_wps, exte_wps, entry_wp):
    graph_inter=nx.Graph()
    graph_exte=nx.Graph()
    graph_global=nx.Graph()

    inter_wps=list(map(tuple, inter_wps))
    exte_wps=list(map(tuple, exte_wps))
    entry_wp=tuple(entry_wp)

    inter_entry_point=((inter_wps[0][0]+inter_wps[-1][0])/2, (inter_wps[0][1]+inter_wps[-1][1])/2)
    inter_wps.append(inter_entry_point)
    exte_entry_point=((exte_wps[0][0]+exte_wps[-1][0])/2, (exte_wps[0][1]+exte_wps[-1][1])/2)
    exte_wps.append(exte_entry_point)

    graph_inter.add_node(inter_wps[0])

    for i in range(1,len(inter_wps)):
        node=inter_wps[i]
        graph_inter.add_node(node)
        graph_inter.add_weighted_edges_from([(inter_wps[i-1],inter_wps[i], dist(inter_wps[i-1], inter_wps[i]))])
    graph_inter.add_weighted_edges_from([(inter_wps[-1],inter_wps[0], dist(inter_wps[-1], inter_wps[0]))])
    
    graph_exte.add_node(exte_wps[0])
    for i in range(1,len(exte_wps)):
        node=exte_wps[i]
        graph_exte.add_node(node)
        graph_exte.add_weighted_edges_from([(exte_wps[i-1],exte_wps[i], dist(exte_wps[i-1], exte_wps[i]))])
    graph_exte.add_weighted_edges_from([(exte_wps[-1],exte_wps[0], dist(exte_wps[-1], exte_wps[0]))])

    graph_global: nx.Graph = nx.union(graph_inter, graph_exte)

    graph_global.add_node(entry_wp)
    graph_global.add_weighted_edges_from([(inter_entry_point, entry_wp, dist(inter_entry_point, entry_wp)),
                                          (exte_entry_point, entry_wp, dist(exte_entry_point, entry_wp))])

    return graph_global


def find_shortest_path(graph: nx.Graph, start_wp_arg, end_wp_arg):
    start_wp=(start_wp_arg[0], start_wp_arg[1])
    end_wp=(end_wp_arg[0], end_wp_arg[1])
    closest_node_from_start=min(graph.nodes, key=lambda node : dist(start_wp, node))
    closest_node_from_end=min(graph.nodes, key=lambda node : dist(end_wp, node))
    graph.add_node(start_wp)
    graph.add_node(end_wp)
    graph.add_weighted_edges_from([(start_wp, closest_node_from_start, dist(start_wp, closest_node_from_start))])
    graph.add_weighted_edges_from([(end_wp, closest_node_from_end, dist(end_wp, closest_node_from_end))])

    path_to_return=nx.astar_path(graph, start_wp, end_wp)
    graph.remove_nodes_from([start_wp, end_wp])
    return path_to_return
    
def compute_shortest_path_length(graph: nx.Graph, start_wp_arg, end_wp_arg):
    start_wp=(start_wp_arg[0], start_wp_arg[1])
    end_wp=(end_wp_arg[0], end_wp_arg[1])
    closest_node_from_start=min(graph.nodes, key=lambda node : dist(start_wp, node))
    closest_node_from_end=min(graph.nodes, key=lambda node : dist(end_wp, node))
    graph.add_node(start_wp)
    graph.add_node(end_wp)
    graph.add_weighted_edges_from([(start_wp, closest_node_from_start, dist(start_wp, closest_node_from_start))])
    graph.add_weighted_edges_from([(end_wp, closest_node_from_end, dist(end_wp, closest_node_from_end))])

    path_length=nx.astar_path_length(graph, start_wp, end_wp, weight='weight')
    graph.remove_nodes_from([start_wp, end_wp])
    return path_length

def create_cost_matrix(blocks: list[Row], agents: list[SimpleAgent]):
    cost_matrix=[]
    closest_row_matrix=[]

    for a in agents:
        cost_matrix_row_to_add=[]
        closest_row_matrix_row_to_add=[]
        
        for b in blocks:
            block_l = sum(row.length for row in b)

            entry_block=[]
            if len(b)>0:
                entry_block.append(b[0]) # first row is an extremity
            if len(b)>1:
                entry_block.append(b[-1]) # last row is an extremity

            dist_to_entry=[]
            for row in entry_block:
                dist_to_entry.append([row.get_extremity_points()[0],
                                      0,
                                      dist(row.get_extremity_points()[0], a.get_starting_point())])
                dist_to_entry.append([row.get_extremity_points()[1],
                                      1,
                                      dist(row.get_extremity_points()[1], a.get_starting_point())])
            if len(entry_block)>0:
                min_dist=min(dist_to_entry, key=lambda x : x[2])

                cost=min_dist[2]+block_l
                cost_matrix_row_to_add.append(cost)
                closest_row_matrix_row_to_add.append(min_dist[:2])
            else:
                cost=0

                cost_matrix_row_to_add.append(cost)
                closest_row_matrix_row_to_add.append([-1,-1])

        cost_matrix.append(cost_matrix_row_to_add)
        closest_row_matrix.append(closest_row_matrix_row_to_add)
    return np.array(cost_matrix), closest_row_matrix

def create_cost_matrix_with_cp(blocks, agents: list[Agent], charging_points, dist_cp_matrix, SAFETY_MARGIN):
    cost_matrix=[]
    agent_matrix=[]
    block_lengths=[]
    entry_rows=[]
    index=0
    for block in blocks:
        b_l= sum(row.length for row in block)
        block_lengths.append(b_l)

        entry_block=[]
        if len(block)>0:
            entry_block.append(block[0]) # first row is an extremity
        if len(block)>1:
            entry_block.append(block[-1]) # last row is an extremity
        entry_rows.append(entry_block)

        index+=1


    for ag in agents:
        cost_matrix_row_to_add=[]
        agent_matrix_row_to_add=[]
        index=0
        for b in blocks:
            agents=[]
            block_id=index
            agent_alloc_list=[]
            for row in entry_rows[block_id]:
                                                
                agent_copy_0=deepcopy(ag)
                agent_copy_1=deepcopy(ag)
                agent_copy_0.non_ordered_rows=blocks[block_id]
                agent_copy_1.non_ordered_rows=blocks[block_id]

                agent_copy_0.entry_point=row.get_extremity_points()[0]
                agent_copy_0.distance_to_entry_point=dist(agent_copy_0.starting_point, agent_copy_0.entry_point)
                agent_copy_0.compute_rows_order()
                res_0=agent_copy_0.compute_optimal_cp(charging_points, dist_cp_matrix, SAFETY_MARGIN)

                agent_copy_1.entry_point=row.get_extremity_points()[1]
                agent_copy_1.distance_to_entry_point=dist(agent_copy_1.starting_point, agent_copy_1.entry_point)
                agent_copy_1.compute_rows_order()
                res_1=agent_copy_1.compute_optimal_cp(charging_points, dist_cp_matrix, SAFETY_MARGIN)

                if res_0==1:
                    agent_copy_0.compute_makespan()
                else:
                    agent_copy_0.makespan=INF
                if res_1==1:
                    agent_copy_1.compute_makespan()
                else:
                    agent_copy_1.makespan=INF
                agent_alloc_list.append(deepcopy(agent_copy_0))
                agent_alloc_list.append(deepcopy(agent_copy_1))


            if agent_alloc_list!=[]:
                agent_to_add=min(agent_alloc_list, key=lambda x : x.makespan)
                cost_matrix_row_to_add.append(agent_to_add.makespan)
                agent_matrix_row_to_add.append(agent_to_add)
            else:
                cost_matrix_row_to_add.append(0)
                agent_matrix_row_to_add.append(None)
            index+=1
        cost_matrix.append(cost_matrix_row_to_add)
        agent_matrix.append(agent_matrix_row_to_add)
    
    return np.array(cost_matrix), agent_matrix

def create_cost_matrix_with_cp_greedy(blocks, agents: list[Agent], charging_points, dist_cp_matrix, SAFETY_MARGIN):
    cost_matrix=[]
    agent_matrix=[]
    block_lengths=[]
    entry_rows=[]
    index=0
    for block in blocks:
        b_l= sum(row.length for row in block)
        block_lengths.append(b_l)

        entry_block=[]
        if len(block)>0:
            entry_block.append(block[0]) # first row is an extremity
        if len(block)>1:
            entry_block.append(block[-1]) # last row is an extremity
        entry_rows.append(entry_block)

        index+=1


    for ag in agents:
        cost_matrix_row_to_add=[]
        agent_matrix_row_to_add=[]
        index=0
        for b in blocks:
            agents=[]
            block_id=index
            agent_alloc_list=[]
            for row in entry_rows[block_id]:
                                                
                agent_copy_0=deepcopy(ag)
                agent_copy_1=deepcopy(ag)
                agent_copy_0.non_ordered_rows=blocks[block_id]
                agent_copy_1.non_ordered_rows=blocks[block_id]

                agent_copy_0.entry_point=row.get_extremity_points()[0]
                agent_copy_0.distance_to_entry_point=dist(agent_copy_0.starting_point, agent_copy_0.entry_point)
                agent_copy_0.compute_rows_order()
                res_0=agent_copy_0.compute_greedy_cp(charging_points, dist_cp_matrix, SAFETY_MARGIN)

                agent_copy_1.entry_point=row.get_extremity_points()[1]
                agent_copy_1.distance_to_entry_point=dist(agent_copy_1.starting_point, agent_copy_1.entry_point)
                agent_copy_1.compute_rows_order()
                res_1=agent_copy_1.compute_greedy_cp(charging_points, dist_cp_matrix, SAFETY_MARGIN)

                if res_0==1:
                    agent_copy_0.compute_makespan()
                else:
                    agent_copy_0.makespan=INF
                if res_1==1:
                    agent_copy_1.compute_makespan()
                else:
                    agent_copy_1.makespan=INF
                agent_alloc_list.append(deepcopy(agent_copy_0))
                agent_alloc_list.append(deepcopy(agent_copy_1))


            if agent_alloc_list!=[]:
                agent_to_add=min(agent_alloc_list, key=lambda x : x.makespan)
                cost_matrix_row_to_add.append(agent_to_add.makespan)
                agent_matrix_row_to_add.append(agent_to_add)
            else:
                cost_matrix_row_to_add.append(0)
                agent_matrix_row_to_add.append(None)
            index+=1
        cost_matrix.append(cost_matrix_row_to_add)
        agent_matrix.append(agent_matrix_row_to_add)
    
    return np.array(cost_matrix), agent_matrix


def create_cost_matrix_with_cp_and_borders(blocks, agents: list[Agent], charging_points, nav_graph):
    cost_matrix=[]

    return cost_matrix

def compute_list_of_agents_makespan(agents: list[Agent]):
    makespan=0
    worst_agent_ind=0
    for ind_ag in range(len(agents)) :
        ag_length=agents[ind_ag].compute_makespan()
        if ag_length>makespan:
            makespan=ag_length
            worst_agent_ind=ind_ag
    return worst_agent_ind, makespan


def give_row(agents: list[Agent], ind_giving_agent, side):

    if agents[ind_giving_agent].non_ordered_rows!=None and len(agents[ind_giving_agent].non_ordered_rows)!=0:
        
        if side==0:
            agents[ind_giving_agent-1].non_ordered_rows.append(agents[ind_giving_agent].non_ordered_rows.pop(0))
            agents[ind_giving_agent-1].compute_entry_point()
            agents[ind_giving_agent-1].compute_rows_order()
            agents[ind_giving_agent].compute_entry_point()
            agents[ind_giving_agent].compute_rows_order()
        elif side==1:
            agents[ind_giving_agent+1].non_ordered_rows.insert(0,agents[ind_giving_agent].non_ordered_rows.pop(-1))
            agents[ind_giving_agent].compute_entry_point()
            agents[ind_giving_agent].compute_rows_order()
            agents[ind_giving_agent+1].compute_entry_point()
            agents[ind_giving_agent+1].compute_rows_order()

def compute_makespan_hungarian(blocks, agents_list):

    cost_matrix, closest_row_matrix=create_cost_matrix(blocks, agents_list)
    # aloc=hungarian_algorithm(cost_matrix)
    m=MunkresMinMax()
    aloc=m.compute(cost_matrix)
    minimal_makespan=aloc[1]
    minimal_robot_allocation=[]
    for a in aloc[0]:
        agent_id=a[0]
        block_id=a[1]
        alloc_to_add=[agents_list[agent_id], blocks[block_id], closest_row_matrix[agent_id][0], closest_row_matrix[agent_id][1]]
        minimal_robot_allocation.append(alloc_to_add)
    return minimal_robot_allocation, minimal_makespan


def compute_makespan_with_cp_hungarian(blocks, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN):

    cost_matrix, agent_matrix=create_cost_matrix_with_cp(blocks, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN)
    m=MunkresMinMax()
    aloc=m.compute(cost_matrix)
    # aloc=hungarian_algorithm(cost_matrix)
    minimal_makespan=aloc[1]
    best_agents=[]
    for a in aloc[0]:
        agent_to_add=agent_matrix[a[0]][a[1]]
        best_agents.append(agent_to_add)
    return best_agents, minimal_makespan


def compute_makespan_with_cp_greedy_hungarian(blocks, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN):

    cost_matrix, agent_matrix=create_cost_matrix_with_cp_greedy(blocks, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN)
    m=MunkresMinMax()
    aloc=m.compute(cost_matrix)
    # aloc=hungarian_algorithm(cost_matrix)
    minimal_makespan=aloc[1]
    best_agents=[]
    for a in aloc[0]:
        agent_to_add=agent_matrix[a[0]][a[1]]
        best_agents.append(agent_to_add)
    return best_agents, minimal_makespan



if __name__ == "__main__":
    import os.path

    xpoints=[2,3,4,5]
    optimal_mean_times=[1.62,3.51,16.93,131.28]
    heuristic_mean_times=[0.02,0.02,0.02,0.03]
    save_path = "/home/simon/these/code/py_rows_allocation/python_ws/energy_studies"
    plt.rcParams.update({'font.size': 21})
    plt.clf()
    plt.figure(figsize=(8, 10), dpi=160)
    plt.plot(xpoints,optimal_mean_times, label='Optimal', linewidth=5)
    plt.plot(xpoints,heuristic_mean_times, label='Heuristic', linewidth=5)
    plt.xlabel("Number of agents")
    plt.ylabel("Time to find solution (seconds) ")
    plt.title("Time to find solution for 50 rows")
    plt.xticks(xpoints, xpoints)
    plt.legend(fontsize="25")
    fig_file_name="cp_time_study_omptimal_vs_heuristic_50rows_seed_2.png" 
    fig_complete_name=os.path.join(save_path, fig_file_name)
    plt.savefig(fig_complete_name)