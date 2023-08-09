'''
Created on June 10, 2021
Implements the Iterative Form of JacSim* for directed and undirected graphs by using multiprocessing.
@author: masoud
'''

import os
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import sharedctypes

link_dict={}
node_set = set ()
keyList = []
graph = ''
result_path = ''
ds_name= ''
decay_factor=0
alpha_val=0
iterations=0
topK=0
link_type=''
last_iteration = None ## --- shared matrix among all cores
current_iteration = None ## --- shared matrix among all cores and manipulated by them
jaccard_scores = None ## 
info_string = ''
total_core_num=0 ## -- keeps the number of CPU cores
    
def read_graph_in_out (graph, link_type):
    '''
        reads graph and make the "in-links" OR "out-links" dictionary.
    '''
    link_dict.clear()
    node_set.clear()
    with open(graph, "r") as f:
        if link_type == 'in-link':
            lines = f.readlines()
            for line in lines:
                node_1, node_2 = line.split('\t')[:2]
                node_set.update((int(node_1),int(node_2)))                        
                if int(node_2) not in link_dict:
                    in_links = set()
                    in_links.add(int(node_1))
                    link_dict [int(node_2)] = in_links
                else:
                    link_dict[int(node_2)].add(int(node_1))                    
        else:
            lines = f.readlines()
            for line in lines:
                node_1, node_2 = line.split('\t')[:2]
                node_set.update((int(node_1),int(node_2)))                        
                if int(node_1) not in link_dict:
                    out_links = set()
                    out_links.add(int(node_2))
                    link_dict [int(node_1)] = out_links
                else:
                    link_dict[int(node_1)].add(int(node_2))
                                                                                 
    f.close()    
    print ('The link dictionary ('+link_type+') is constructed ... \n')    
    
def initializatoin (graph_='', decay_factor_=0, alpha_=1.0, iterations_=0, link_type_='', total_core_num_=0):
    '''
        initialize the required variables and memories for all process.
    '''    

    global graph
    global result_path
    global ds_name
    global decay_factor
    global alpha_val
    global iterations
    global topK
    global last_iteration
    global current_iteration
    global jaccard_scores    
    global info_string
    global node_set
    global keyList
    global total_core_num
    global link_dict
    global link_type

    graph = graph_
    decay_factor = decay_factor_
    alpha_val = alpha_
    iterations = iterations_
    link_type = link_type_
    total_core_num = total_core_num_   

    if link_type not in {'in-link', 'out-link','none'}:
        print('Link-type must be in-link, out-link, or none ...')
        return        
    print("JacSim* Iterative Form (multi-processing) is Started ....... ")
    #============================================================================================
        # reading graph; constructing the link set of each node
    #============================================================================================
    read_graph_in_out(graph, link_type)    
    # =================================================================================
        # defines a matrix that is SHARED among all cores
    # =================================================================================
    last_iteration_base = np.ctypeslib.as_ctypes(np.identity((len(node_set)))) ## set S(a,a)=1
    last_iteration = sharedctypes.RawArray(last_iteration_base._type_, last_iteration_base)    
    jaccard_scores_base = np.ctypeslib.as_ctypes(np.zeros((len(node_set),len(node_set))))
    jaccard_scores = sharedctypes.RawArray(jaccard_scores_base._type_, jaccard_scores_base) # keeps the ORIGINAL Jaccard scores for all node-pairs
    node_set.clear()
    keyList = list (link_dict)    
    print ('Number of nodes with neighbors: '+str(len(keyList)))
    
    sum_of_neghbors = 0
    for i in range (0, len(keyList)):
        sum_of_neghbors = sum_of_neghbors + len(link_dict[keyList[i]])
    print ('Average number of neighbors per node: '+str(sum_of_neghbors/len(keyList))+'\n')
        
    non_empty_neighb_intersec = 0
    #============================================================================================
        # Computing Jaccard scores
        # Instead of computing the similarity for all node-pairs in the graph, 
        # we do computation only for those nodes having neighbors.        
    #============================================================================================
    print('Iteration 1 \n')        
    for target_node_index in range (0,len(keyList)):
        target_node = keyList[target_node_index]        
        for node_index in range (target_node_index+1,len(keyList)):       # s(i,i) is NOT computed; it was set as 1 when creating last_iteration 
            node = keyList[node_index]                    
            intersection = link_dict[target_node].intersection(link_dict[node])
            intersection_size = len(intersection)
            if intersection_size!=0:
                non_empty_neighb_intersec = non_empty_neighb_intersec + 1
                union_size = len(link_dict[target_node].union(link_dict[node]))                
                jaccard_scores[target_node][node] = jaccard_scores[node][target_node] = intersection_size/float (union_size)
                '''
                    iteration 1 is equal to original Jaccard scores effected by alpha and decay factor    
                '''
                last_iteration[target_node][node] = last_iteration[node][target_node] = (decay_factor*alpha_)*(intersection_size/float (union_size))
                            
    last_iteration = np.ctypeslib.as_array(last_iteration)
    print('# of nodes with non-empty common neighbor set: '+ str(non_empty_neighb_intersec))    
    
    # ==============================================================================================
        # computation for K>1
    # ==============================================================================================
    for itr in range (2, iterations+1):     
        print("Iteration "+str(itr)+'\n') 
        current_iteration_base = np.ctypeslib.as_ctypes(np.zeros( (len(last_iteration),len(last_iteration))))         
        current_iteration = sharedctypes.RawArray(current_iteration_base._type_, current_iteration_base)
        jobs = []                    
        for i in range(total_core_num):
            p = Process(target = JacSim_Star, args = (i,))
            jobs.append(p)
            p.start()
        for i in range(total_core_num):
            jobs[i].join()                    
        current_iteration = np.ctypeslib.as_array(current_iteration) ## we need to change the ctypes-array as a np-array to apply np-array's related operations
        np.fill_diagonal(current_iteration,1)  ## diagonal entries again are set as 1 
        last_iteration = current_iteration
    return current_iteration
       
def JacSim_Star (core_number):
    for target_node_index in range (0+core_number,len(keyList),total_core_num):
        target_node = keyList[target_node_index]
        for node_index in range (target_node_index+1,len(keyList)):       # s(i,i) NOT is computed
            node = keyList[node_index]
            target_neighbors = list(link_dict[target_node])
            node_neighbors = list(link_dict[node])
            sum_ = 0
            for target_node_neighbor in target_neighbors:                    
                for node_neighbor in node_neighbors:
                    if target_node_neighbor!= node_neighbor: ## only (i,j) where i!=j (i.e., not i,j IN INTERSECTION where i==j)
                        sum_ = sum_ + last_iteration[target_node_neighbor][node_neighbor]
            '''
                if sum_ = 0 means 
                    1) target_node and node have only one single in-link that is identical
                    2) similarity scores of all possible inlink-pairs from last iteration are zero 
            '''                            
            if sum_!=0:
                current_iteration[target_node][node] = current_iteration[node][target_node] = decay_factor * ( alpha_val* jaccard_scores[target_node][node] + (1.0-alpha_val) * sum_/(len(target_neighbors)*len(node_neighbors) ))
            else:
                current_iteration[target_node][node] = current_iteration[node][target_node] = decay_factor * alpha_val * jaccard_scores[target_node][node]                                 
