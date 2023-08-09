'''
Created on Jun 15, 2021
Implements the Matrix Form of JacSim* for directed and undirected graphs.
@author: masoud
'''
import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize

def compute_JacSimStar (graph='', decay_factor=0, iterations=0, alpha_val=1.0,link_type=''):
            
    if link_type not in {'in-link', 'out-link','none'}:
        print('Link-type must be in-link, out-link, or none ...')
        return
    print("JacSim* Matrix Form is Started ....... ")    
    #============================================================================================
        # reading graph, computing Jaccard scores
    #============================================================================================    
    G = nx.read_edgelist(graph, create_using=nx.DiGraph(), nodetype = int)    
    adj = nx.adjacency_matrix(G,nodelist=sorted(G.nodes()), weight=None)      # V*V adjacency matrix
    J = np.ones((len(G.nodes()),len(G.nodes()))) # it is a |V|*|V| all-ones matrix (all elements are one)        
    
    if link_type =='in-link':
        U = adj.T*J + (adj.T*J).T ## |I_a|+|I_b|
        U = 1/(U-(adj.T*adj)) ## An element-wise division; We compute (1/UNION-INTERSECT)
        print("Union matrix U is created ...")    
        jaccard_scores = (adj.T*adj).multiply(U) ## to prevent NAN values, instead of (INTERSECT/UNION-INTERSECT), we do INTERSECT*(1/UNION-INTERSECT)
        print("Jaccard scores are calculated ...")
        norm_adj = normalize(adj, norm='l1', axis=0) # column normalized adjacency matrix
        print("Column normalized adjacency matrix constructed ...")
             
    else: ## for out-link or none
        U = adj*J + (adj*J).T
        U = 1/(U-(adj*adj.T)) ## We compute (1/UNION-INTERSECT)
        print("Union matrix U is created ...")    
        jaccard_scores = (adj*adj.T).multiply(U) ## to prevent NAN values, instead of (INTERSECT/UNION-INTERSECT), we do INTERSECT*(1/UNION-INTERSECT)
        print("Jaccard scores are calculated ...")
        norm_adj = normalize(adj, norm='l1', axis=1) # row normalized adjacency matrix
        print("Row normalized adjacency matrix constructed ...")        
    print('================================================================================================================================')
    
    #============================================================================================
        # JacSim* for k=1
    #============================================================================================         
    print('Iteration 1 ....\n')        
    result_matrix = decay_factor*alpha_val*jaccard_scores
    result_matrix = result_matrix.todense() 
    np.fill_diagonal(result_matrix,1) ## set diagonal values to one; it is just for writing results since they are NOT used in computing similarity scores                                            
            
    for itr in range (2, iterations+1):
        print("Iteration "+str(itr)+' .... \n')
        np.fill_diagonal(result_matrix,0)
        if link_type =='in-link': 
            result_matrix =  decay_factor * (alpha_val* jaccard_scores + (1-alpha_val) * (norm_adj.T * result_matrix * norm_adj)) #+ iden_matrix
        else: ## for out-link or none
            result_matrix =  decay_factor * (alpha_val* jaccard_scores + (1-alpha_val) * (norm_adj * result_matrix * norm_adj.T)) #+ iden_matrix
        np.fill_diagonal(result_matrix,1)
    print('JasSin* computation is completed ...\n') 
    return result_matrix
