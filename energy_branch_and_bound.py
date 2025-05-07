from numpy import inf as INF
import matplotlib.pyplot as plt
import random
import time
from queue import PriorityQueue
from Node import NodeWithCP
from Utils import LineSeg, Row, Agent, Traveling, Charging
from energy_heuristic import random_charging_points, compute_dist_cp_matrix, patches, draw_semicircle, colors, lines, Heuristic_cp
from field_and_rows import to_convex_contour, rows_creator_nb_rows, random_starting_points

class OptimalSearchCP():
    def __init__(self, line_pts, agents_list, charging_points, timeout_time, h_makespan_slow_depth, h_prio_mode=1):


        self.timeout_time=timeout_time
        self.line_pts=line_pts

        linesegs = [LineSeg(line_pts[i][0], line_pts[i][1]) for i in range(len(line_pts))]
        self.lengths = [lineseg.length() for lineseg in linesegs]

        self.rows_list=[]
        self.fill_rows_list()

        self.charging_points=charging_points
        self.dist_cp_matrix=compute_dist_cp_matrix(charging_points, line_pts)

        self.agents_list=agents_list
        self.nb_nodes=1
        self.prioQ: PriorityQueue[NodeWithCP] = PriorityQueue()
        self.upper_bound=INF
        self.init_upper_bound=INF
        self.best_agents=None

        self.start_h=None
        self.end_h=None

        self.h_makespan_slow_depth = h_makespan_slow_depth 
        self.h_prio_mode=h_prio_mode # 1 : h_prio everytime, 2 : 0 when leaf


    def get_optimal_solution(self):
         return self.best_agents, self.upper_bound, self.init_upper_bound, (self.end_h-self.start_h)

    def fill_rows_list(self):
            id_row=0
            for r in self.line_pts:
                row=Row(id_row, r, self.lengths[id_row], self.lengths[id_row])
                id_row+=1
                self.rows_list.append(row)
                

    def find_optimal(self):


        nb_agents=len(self.agents_list)

        self.start_h=time.time()
        agents, self.upper_bound  = Heuristic_cp(self.agents_list, line_pts, starting_points, charging_points)
        self.end_h=time.time()


        self.init_upper_bound=self.upper_bound
        # print("Initial UB : ", self.init_upper_bound)
        first_node=NodeWithCP(0, 0, None, None, 0)
        self.prioQ.put(first_node)
        average=sum([row.length for row in self.rows_list])/nb_agents

        if self.timeout_time==None:
            while not self.prioQ.empty():
                node_to_open=self.prioQ.get()

                res=node_to_open.open(self.rows_list, self.agents_list, self.nb_nodes, self.upper_bound, average, self.prioQ, self.h_makespan_slow_depth, self.h_prio_mode, self.charging_points, self.dist_cp_matrix, SAFETY_MARGIN)
                
                if res!=1 and res[1]<=self.upper_bound:
                    self.upper_bound=res[1]
                    self.best_agents=res[0]
                    # print("New best makespan found : ", self.upper_bound)
            # print("Optimal Makespan : ", self.upper_bound)
            return 1

        else:
            while not self.prioQ.empty() and time.time()<self.timeout_time:
                node_to_open=self.prioQ.get()

                res=node_to_open.open(self.rows_list, self.agents_list, self.nb_nodes, self.upper_bound, average, self.prioQ, self.h_makespan_slow_depth, self.h_prio_mode, self.charging_points, self.dist_cp_matrix, SAFETY_MARGIN)

                if res!=1 and res[1]<=self.upper_bound:
                    self.upper_bound=res[1]
                    self.best_allocation=res[0]
                    # print("New best makespan found : ", self.upper_bound)
            
            if time.time()<self.timeout_time:
                is_opti=True
            else:
                is_opti=False
            
            return is_opti
            
        # print("Optimal solution found ! ")

class OptimalSearchCPEvo():
    def __init__(self, line_pts, agents_list, charging_points, timeout_time, h_makespan_slow_depth, h_prio_mode=1):


        self.timeout_time=timeout_time
        self.evo=[]
        self.line_pts=line_pts

        linesegs = [LineSeg(line_pts[i][0], line_pts[i][1]) for i in range(len(line_pts))]
        self.lengths = [lineseg.length() for lineseg in linesegs]

        self.rows_list=[]
        self.fill_rows_list()

        self.charging_points=charging_points
        self.dist_cp_matrix=compute_dist_cp_matrix(charging_points, line_pts)

        self.agents_list=agents_list
        self.nb_nodes=1
        self.prioQ: PriorityQueue[NodeWithCP] = PriorityQueue()
        self.upper_bound=INF
        self.init_upper_bound=INF
        self.best_agents=None

        self.start_h=None
        self.end_h=None

        self.h_makespan_slow_depth = h_makespan_slow_depth 
        self.h_prio_mode=h_prio_mode # 1 : h_prio everytime, 2 : 0 when leaf


    def get_optimal_solution(self):
         return self.best_agents, self.upper_bound, self.init_upper_bound, (self.end_h-self.start_h)

    def fill_rows_list(self):
            id_row=0
            for r in self.line_pts:
                row=Row(id_row, r, self.lengths[id_row], self.lengths[id_row])
                id_row+=1
                self.rows_list.append(row)
                

    def find_optimal(self):


        nb_agents=len(self.agents_list)

        self.start_h=time.time()
        agents, self.upper_bound  = Heuristic_cp(self.agents_list, line_pts, starting_points, charging_points)
        self.end_h=time.time()
        self.evo.append((time.time(),self.upper_bound))


        self.init_upper_bound=self.upper_bound
        # print("Initial UB : ", self.init_upper_bound)
        first_node=NodeWithCP(0, 0, None, None, 0)
        self.prioQ.put(first_node)
        average=sum([row.length for row in self.rows_list])/nb_agents

        if self.timeout_time==None:
            while not self.prioQ.empty():
                node_to_open=self.prioQ.get()

                res=node_to_open.open(self.rows_list, self.agents_list, self.nb_nodes, self.upper_bound, average, self.prioQ, self.h_makespan_slow_depth, self.h_prio_mode, self.charging_points, self.dist_cp_matrix, SAFETY_MARGIN)
                
                if res!=1 and res[1]<=self.upper_bound:
                    self.upper_bound=res[1]
                    self.best_agents=res[0]
                    self.evo.append((time.time(),self.upper_bound))

            return self.evo

        else:
            while not self.prioQ.empty() and time.time()<self.timeout_time:
                node_to_open=self.prioQ.get()

                res=node_to_open.open(self.rows_list, self.agents_list, self.nb_nodes, self.upper_bound, average, self.prioQ, self.h_makespan_slow_depth, self.h_prio_mode, self.charging_points, self.dist_cp_matrix, SAFETY_MARGIN)

                if res!=1 and res[1]<=self.upper_bound:
                    self.upper_bound=res[1]
                    self.best_allocation=res[0]
                    # print("New best makespan found : ", self.upper_bound)
            
            if time.time()<self.timeout_time:
                is_opti=True
            else:
                is_opti=False
            
            return is_opti
            
       

NB_SIDE_POLYGON=8
NB_ROWS=30
NB_AGENTS=4
NB_CHARGING_POINTS=10
MAX_CHARGE=40
SAFETY_MARGIN=0.05

if __name__ == "__main__":

    rd_seed=15
    points = to_convex_contour(NB_SIDE_POLYGON, rd_seed)

    # border_with_entry, entry_point, inter_waypoints, inter_entry_point, exter_waypoints, exter_entry_point = create_border(array(points), FIELD_SPACE_MARGING)

    line_pts=rows_creator_nb_rows(points, NB_ROWS)
    starting_points = random_starting_points(NB_AGENTS, rd_seed)

    id=0
    agents_list=[]
    for sp in starting_points:
        init_charge=random.randint(int(MAX_CHARGE/3),MAX_CHARGE)
        ag=Agent(id, sp, init_charge, MAX_CHARGE, None, None, None)
        agents_list.append(ag)
        id+=1
    charging_points=random_charging_points(NB_CHARGING_POINTS)

    before=time.time()
    optimal_search=OptimalSearchCP(line_pts, agents_list, charging_points, None, 5, 1 )
    is_opti=optimal_search.find_optimal()
    optimal_alloaction, makespan,init_upper_bound, delay=optimal_search.get_optimal_solution()
    after=time.time()

    if optimal_alloaction!=None:
        print("\nOptimal allocation : ")
        for ag in optimal_alloaction:
            print("Agent "+ str(ag.get_id())+" : ", ag.rows)
        print("\nMakespan = ", makespan)
        print("Time (in seconds) to find solution : ",after-before)
        ## We create the final fig ##
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Optimal solution')
        polygon = patches.Polygon(points, closed = True, fill = False)
        ax.add_artist(polygon)

        for ag in optimal_alloaction :

            color = colors[ag.id]
            ax.plot(ag.starting_point[0], ag.starting_point[1], marker="X", markersize=8, color=color)
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
                            if ind%2==1:
                                draw_semicircle(x1, y1, x2, y2, color=color, lw=1, ax=ax)
                            else:
                                draw_semicircle(x2, y2, x1, y1, color=color, lw=1, ax=ax)
                        else:
                            if ind%2==0:
                                draw_semicircle(x1, y1, x2, y2, color=color, lw=1, ax=ax)
                            else:
                                draw_semicircle(x2, y2, x1, y1, color=color, lw=1, ax=ax)
                        ind+=1
                        # ax_o.add_artist(lines.Line2D([stage.start_pt[0], stage.end_pt[0]],[stage.start_pt[1], stage.end_pt[1]], color=color))
                elif type(stage)==type(Charging(None, None, None)):
                    ind+=1
                    # ax.annotate(str(int(stage.energy_on_arrival*100)/100)+"->"+str(stage.energy_on_leaving),xy=(stage.pos[0], stage.pos[1]+0.5))

        for cp in charging_points :
            ax.plot(cp[0], cp[1], marker="s", color="gray")
        ax.set_xlim(-6, 11)
        ax.set_ylim(-6, 11)
        print("Makespan : ",end="")
        print(makespan)
        plt.show()
    else:
        print("No possible solution.")
