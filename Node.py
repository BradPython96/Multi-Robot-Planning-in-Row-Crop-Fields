import numpy as np
from itertools import permutations
from Utils import Row, Agent, SimpleAgent
from math import dist
from queue import PriorityQueue
from dataclasses import dataclass
from typing import Any
from copy import copy, deepcopy
from Utils import compute_makespan_hungarian, compute_makespan_with_cp_hungarian
INF=np.inf

@dataclass(order=True)
class Node():
    def __init__(self, id, prio, block, pred, depth):

        self.id=id
        self.priority=prio
        self.pred: Node = pred  
        self.block: list[Row] = block  
        self.depth = depth # Starting with number 1


    def __del__(self):
        if 0:
            print("Node "+str(self.id)+" removed.")
    
    def __repr__(self) -> str:
        typename = type(self).__name__
        return f'{typename}(id={self.id}, priority={self.priority}, depth={self.depth}, {self.block})'
    
    def get_id(self):
         return self.id
    
    def get_pred(self):
        return self.pred
    
    def get_block(self):
        return self.block

    def get_priority(self):
        return self.priority

    def open(self, 
            rows_list: list[Row], 
            agents_list: list[Agent], 
            last_node_id: int, 
            UB: float,
            average: float,
            prioQ: PriorityQueue,
            h_makespan_slow_depth : int,
            h_prio_mode : int

        ):


        if self.depth<len(agents_list)-1:

            if self.get_block()==None or self.get_block()==[]:
                for end_index in range(0, len(rows_list)+1):
                    block=[row for row in rows_list[:end_index]]


                    if self.depth+1<=h_makespan_slow_depth:
                        LB=h_makespan_slow(block,agents_list)
                    else:
                        LB=h_makespan_fast(block)

                    if LB <= UB:
                        prio=h_priority(block, average)
                        node_to_add=Node(last_node_id, prio, block, self, self.depth+1)

                        prioQ.put(node_to_add)
                        last_node_id+=1
    
            else:
                first_row_id=self.get_block()[-1].get_id()+1

                last_row_id=first_row_id+1
                end=len(rows_list)

                while last_row_id!=end:
                    block=rows_list[first_row_id:last_row_id]
                    last_row_id+=1

                    if self.depth+1<=h_makespan_slow_depth:
                        LB=h_makespan_slow(block,agents_list)
                    else:
                        LB=h_makespan_fast(block)

                    if LB <= UB:
                        prio=h_priority(block, average)
                        node_to_add=Node(last_node_id, prio, block, self, self.depth+1)
                        prioQ.put(node_to_add)
                        last_node_id+=1
            return 1

        elif self.depth==len(agents_list)-1:
            if self.get_block()==[]:
                block=rows_list
            else:
                first_row_id=self.get_block()[-1].get_id()+1
                block=rows_list[first_row_id:]
                
            if self.depth+1<=h_makespan_slow_depth:
                LB=h_makespan_slow(block,agents_list)
            else:
                LB=h_makespan_fast(block)

            if LB <= UB:
                if h_prio_mode==1:
                    prio=h_priority(block, average)
                elif h_prio_mode==2:
                    prio=0
                node_to_add=Node(last_node_id, prio, block, self, self.depth+1)

                prioQ.put(node_to_add)
                last_node_id+=1
            
            return 1

        return self.return_makespan(agents_list)

    def return_makespan(self, agents_list):
        pred=self.get_pred()
        blocks=[]
        blocks.append(self.block)

        while pred.get_block()!=None:
            blocks.append(pred.get_block())
            pred=pred.get_pred()

        # alloc, makespan= compute_makespan(blocks, agents_list)
        alloc, makespan=compute_makespan_hungarian(blocks, agents_list)

        # if hung_makespan!=makespan:
        #     print(hung_makespan, makespan)
        for al in alloc:
            al[0].rows=al[1]
            al[0].non_ordered_rows=sorted(copy(al[1]), key=lambda x: x.get_id())
            al[0].compute_entry_point()
            al[0].compute_rows_order()
            al[0].remove_actions()
            if len(al[1])>0:
                al[0].create_actions()

        return agents_list, makespan
    
@dataclass(order=True)
class NodeWithCP():
    def __init__(self, id, prio, block, pred, depth):

        self.id=id
        self.priority=prio
        self.pred: Node = pred  
        self.block: list[Row] = block  
        self.depth = depth # Starting with number 1


    def __del__(self):
        if 0:
            print("Node "+str(self.id)+" removed.")
    
    def __repr__(self) -> str:
        typename = type(self).__name__
        return f'{typename}(id={self.id}, priority={self.priority}, depth={self.depth}, {self.block})'
    
    def get_id(self):
         return self.id
    
    def get_pred(self):
        return self.pred
    
    def get_block(self):
        return self.block

    def get_priority(self):
        return self.priority

    def open(self, 
            rows_list: list[Row], 
            agents_list: list[Agent], 
            last_node_id: int, 
            UB: float,
            average: float,
            prioQ: PriorityQueue,
            h_makespan_slow_depth : int,
            h_prio_mode: int,
            charging_points: list,
            dist_cp_matrix: list,
            SAFETY_MARGIN: float
        ):


        if self.depth<len(agents_list)-1:

            if self.get_block()==None or self.get_block()==[]:
                for end_index in range(0, len(rows_list)+1):
                    block=[row for row in rows_list[:end_index]]


                    if self.depth+1<=h_makespan_slow_depth:
                        LB=h_makespan_slow(block,agents_list)
                    else:
                        LB=h_makespan_fast(block)

                    if LB <= UB:
                        prio=h_priority(block, average)
                        node_to_add=NodeWithCP(last_node_id, prio, block, self, self.depth+1)

                        prioQ.put(node_to_add)
                        last_node_id+=1
    
            else:
                first_row_id=self.get_block()[-1].get_id()+1

                last_row_id=first_row_id+1
                end=len(rows_list)

                while last_row_id!=end:
                    block=rows_list[first_row_id:last_row_id]
                    last_row_id+=1

                    if self.depth+1<=h_makespan_slow_depth:
                        LB=h_makespan_slow(block,agents_list)
                    else:
                        LB=h_makespan_fast(block)

                    if LB <= UB:
                        prio=h_priority(block, average)
                        node_to_add=NodeWithCP(last_node_id, prio, block, self, self.depth+1)
                        prioQ.put(node_to_add)
                        last_node_id+=1
            return 1

        elif self.depth==len(agents_list)-1:
            if self.get_block()==[]:
                block=rows_list
            else:
                first_row_id=self.get_block()[-1].get_id()+1
                block=rows_list[first_row_id:]
                
            if self.depth+1<=h_makespan_slow_depth:
                LB=h_makespan_slow(block,agents_list)
            else:
                LB=h_makespan_fast(block)

            if LB <= UB:
                if h_prio_mode==1:
                    prio=h_priority(block, average)
                elif h_prio_mode==2:
                    prio=0
                node_to_add=NodeWithCP(last_node_id, prio, block, self, self.depth+1)

                prioQ.put(node_to_add)
                last_node_id+=1
            
            return 1

        return self.return_makespan(agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN)

    def return_makespan(self, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN):
        pred=self.get_pred()
        blocks=[]
        blocks.append(self.block)

        while pred.get_block()!=None:
            blocks.append(pred.get_block())
            pred=pred.get_pred()

        # best_agents, makespan= compute_makespan_with_cp(blocks, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN)
        best_agents, makespan= compute_makespan_with_cp_hungarian(blocks, agents_list, charging_points, dist_cp_matrix, SAFETY_MARGIN)

        return best_agents, makespan
    
def h_makespan_fast(block):
    return sum([row.length for row in block])

def h_makespan_slow(block: list[Row], agents: list[SimpleAgent]):

    if len(block)==0:
        return 0
    
    dist_list=[]
    if len(block)==1:
        for ag in agents:
            sp=ag.get_starting_point()
            dist_1=dist(sp, block[0].get_extremity_points()[0])
            dist_2=dist(sp, block[0].get_extremity_points()[1])
            dist_list.append(dist_1)
            dist_list.append(dist_2)
        return sum([row.length for row in block])+min(dist_list)     

    for ag in agents:
        sp=ag.get_starting_point()
        dist_1=dist(sp, block[0].get_extremity_points()[0])
        dist_2=dist(sp, block[0].get_extremity_points()[1])
        dist_3=dist(sp, block[-1].get_extremity_points()[0])
        dist_4=dist(sp, block[-1].get_extremity_points()[1])
        dist_list.append(dist_1)
        dist_list.append(dist_2)
        dist_list.append(dist_3)
        dist_list.append(dist_4)

    return sum([row.length for row in block])+min(dist_list)

def h_priority(block, average):
     block_length=sum([row.length for row in block])
     return int(abs(block_length-average))

def compute_makespan(blocks, agents_list):
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

    minimal_makespan=INF
    minimal_robot_allocation=[]

    iter_robot_block = permutations(agents_list)

    minimal_makespan=INF
    for it in iter_robot_block:
        index=0
        max_makespan=0
        alloc=[]
        for agent in it:
            block_id=index
            alloc_to_add=[agent, blocks[block_id]]
            min_dist=INF
            for row in entry_rows[block_id]:

                dist_0=dist(row.get_extremity_points()[0], agent.get_starting_point())
                dist_1=dist(row.get_extremity_points()[1], agent.get_starting_point())
                
                if dist_0<=dist_1 and dist_0<min_dist:
                    entry_row=row
                    entry_p_ind=0
                    min_dist=dist_0
                        
                elif dist_1<min_dist:
                    entry_row=row
                    entry_p_ind=1
                    min_dist=dist_1
            if min_dist==INF:
                makespan=0
            else:
                makespan=min_dist+block_lengths[block_id]
                alloc_to_add+=[entry_row, entry_p_ind]
            alloc.append(alloc_to_add)

            if makespan>max_makespan:
                max_makespan=makespan
            index+=1

        if max_makespan<minimal_makespan:
            minimal_makespan=max_makespan
            minimal_robot_allocation=alloc

    return minimal_robot_allocation, minimal_makespan

def compute_makespan_with_cp(blocks:list[list[Row]], agents_list:list[Agent], charging_points, dist_cp_matrix, SAFETY_MARGIN):

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

    minimal_makespan=INF
    best_agents=[]

    iter_robot_block = permutations(agents_list)

    minimal_makespan=INF
    for it in iter_robot_block:
        index=0
        max_makespan=0
        agents=[]
        for agent in it:
            block_id=index
            agent_alloc_list=[]
            for row in entry_rows[block_id]:
                                                
                agent_copy_0=deepcopy(agent)
                agent_copy_1=deepcopy(agent)
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
                agents.append(agent_to_add)

                if agent_to_add.makespan>max_makespan:
                    max_makespan=agent_to_add.makespan
                index+=1


        if max_makespan<minimal_makespan:
            minimal_makespan=max_makespan
            best_agents=agents

    return best_agents, minimal_makespan



