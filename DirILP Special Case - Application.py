import glob, os
import matplotlib.pyplot as plt
import pyarrow.parquet as pq  # noqa # READING WITH PYARROW #############
import math
import itertools
from math import log as ln
from math import exp
import pandas as pd
import numpy as np
from gurobipy import *
from scipy import spatial
from scipy import sparse
from collections import Counter
from multiprocessing import Pool
import pickle

filename = os.path.basename(__file__)

if not os.path.exists('/home/idies/workspace/Storage/tunguyen52/Nway/Data/log/' + filename):
    os.makedirs('/home/idies/workspace/Storage/tunguyen52/Nway/Data/log/' + filename)

def get_candidate_list(clustering_result, catalog_label, clustering_label):
    candidate_dict = {}
    list_of_source_index = np.where(clustering_result.labels_==clustering_label) #Find the index in X of all sources with label = clustering_label
    list_of_label = catalog_label[list_of_source_index] #The catalog id of all sources found in the island
    for j in np.unique(list_of_label):
        list_of_indices = []
        for k in range(sum(list_of_label==j)):
            list_of_indices.append(((np.where(cmat[j][:,:2]==X[list_of_source_index][list_of_label==j][k])[0][0]), j))
        candidate_dict[j] = list_of_indices # catalog id: (source id, catalog id)

    # Get all the (catalog_id, source_id) pairs
    candidate_list = []
    for key in candidate_dict.keys():
        list_of_value = candidate_dict[key]
        for pair in list_of_value:
            candidate_list.append(pair)

    catalog_list = []
    source_list = []
    coord_1_list = []
    coord_2_list = []
    Sigma_list = []
    for catalog_source_pair in candidate_list:
        catalog_list.append(catalog_source_pair[1])
        source_list.append(catalog_source_pair[0])
        coord_1_list.append(cmat[catalog_source_pair[1]][catalog_source_pair[0]][0])
        coord_2_list.append(cmat[catalog_source_pair[1]][catalog_source_pair[0]][1])
        Sigma_list.append(cmat[catalog_source_pair[1]][catalog_source_pair[0]][2])
    df = pd.DataFrame({'Source_id': source_list, 'Catalog_id': catalog_list, 'Coordinate_1': coord_1_list, 'Coordinate_2': coord_2_list}) #Make a dataframe from the information provided
    return(candidate_list, df, catalog_list, Sigma_list)

def get_distance(list_of_indices, df):
    '''
    Given 2 pairs of (source_index, catalog_index), return the square distance between them
    df has the columns ('Catalog id', 'Source id', Coord 1, Coord 2, Sigma)
    '''
    coord_list = []
    for i in range(2):
        coord_list+=[np.array(df[(df['Catalog_id'] == list_of_indices[i][1]) & (df['Source_id'] == list_of_indices[i][0])].iloc[:,-2:])[0]]
    array_matrix = np.array(coord_list)
    return(np.linalg.norm(array_matrix[1]-array_matrix[0])**2)

def sum_of_distance(list_of_objects, df):
    '''
    Given n pairs of (source_index, catalog_index), return the sum of all pairwise square distance.
    '''
    num_of_objects = len(list_of_objects)
    coord_list = []
    for i in range(num_of_objects):
        coord_list+=[np.array(df[(df['Catalog_id'] == list_of_objects[i][1]) & (df['Source_id'] == list_of_objects[i][0])].iloc[:,-2:])[0]]
    array_matrix = np.array(coord_list)
    pairwise_dist = spatial.distance.pdist(np.array(array_matrix))**2
    sum_of_square_dist = sum(pairwise_dist)
    return sum_of_square_dist

def Bayes_factor(list_of_objects, df):
    '''
    Compute -ln B_o
    '''
    sum_ln_kappa_rad = 0
    kappa_rad_sum = 0
    kappa_sum = 0
    neg_ln_Bayes = 0
    num_of_objects = len(list_of_objects)
    for object in list_of_objects:
        sum_ln_kappa_rad += ln(kappa_rad_dict[object])
        kappa_rad_sum += kappa_rad_dict[object]
        kappa_sum += kappa_dict[object]
    for index_1 in range(num_of_objects):
        for index_2 in range(index_1+1,num_of_objects):
            neg_ln_Bayes+=(1/(4*kappa_sum))*kappa_dict[list_of_objects[index_1]]*kappa_dict[list_of_objects[index_2]]*get_distance([list_of_objects[index_1],list_of_objects[index_2]],df)
    neg_ln_Bayes = neg_ln_Bayes + (1 - num_of_objects)*ln(2) - sum_ln_kappa_rad + ln(kappa_rad_sum)
    return(neg_ln_Bayes)

def compute_distance_dictionary(list_of_indices, df):
    '''
    Return a dictionary with the form: dict[('Source_id_1', 'Catalog_id_1'), ('Source_id_2', 'Catalog_id_2')] = square distance between them. 
    '''
    distance_dict = {}
    for current_pair_index in range(len(list_of_indices)):
        for next_pair_index in range(current_pair_index + 1, len(list_of_indices)):
            if list_of_indices[next_pair_index][1]!= list_of_indices[current_pair_index][1]: # Only find distances for sources from different catalogs
                distance_dict[(list_of_indices[current_pair_index],list_of_indices[next_pair_index])] = get_distance([list_of_indices[current_pair_index],list_of_indices[next_pair_index]], df)
    return distance_dict

start_time = [0]
def mycallback(model, where):
    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(GRB.Callback.PRE_COLDEL) #number of cols removed by presolve to this point
        rdels = model.cbGet(GRB.Callback.PRE_ROWDEL) #number of rows removed by presolve to this point
    elif where == GRB.Callback.SIMPLEX: #Currently in simplex
        # Simplex callback
        itcnt = model.cbGet(GRB.Callback.SPX_ITRCNT) #Current simplex iteration count
        #if itcnt - model._lastiter >= 100:
        model._lastiter = itcnt
        obj = model.cbGet(GRB.Callback.SPX_OBJVAL) # Current simplex objective value
        ispert = model.cbGet(GRB.Callback.SPX_ISPERT) 
        pinf = model.cbGet(GRB.Callback.SPX_PRIMINF) # Current primal infeasibility
        dinf = model.cbGet(GRB.Callback.SPX_DUALINF) # Current dual infeasibility
        if ispert == 0:
            ch = ' '
        elif ispert == 1:
            ch = 'S'
        else:
            ch = 'P'
    elif where == GRB.Callback.MIP: #Currently in MIP
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT) #Current explored node count
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST) #Current best objective
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND) #Current best objective bound
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT) #Current count of feasible solutions found
        model._lastnode = nodecnt
        actnodes = model.cbGet(GRB.Callback.MIP_NODLFT) #Current unexplored node count
        itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT) #Current simplex iteration count
        cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT) #Current count of cutting planes applied
        if model.cbGet(GRB.Callback.RUNTIME) - start_time[-1] > 2700: #Set a time limit of 45 minutes between feasible solutions
            print('Stop early - Time limit achieved')
            model.terminate()
    elif where == GRB.Callback.MIPSOL: #Found a new MIP incumbent
        # MIP solution callback
        start_time.append(model.cbGet(GRB.Callback.RUNTIME)) 
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT) #Current explored node count
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ) #Objective value for new solution
        objbst = model.cbGet(GRB.Callback.MIPSOL_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT) #Curent count of feasible solutions found
        best_gap = abs(100*(objbnd - objbst)/objbst)
        current_gap = abs(100*(objbnd - obj)/obj) #Gap of the new solution found
        x_MIP = model.cbGetSolution(model._vars)
        x_MIP = np.array(x_MIP)
        index_list = np.where(x_MIP > 0.1)
        res_list = [model._vars[i] for i in index_list[0]]
        nonzero_x = []
        for solution in res_list:
            if solution.Varname[1:4] == "'x'":
                nonzero_x.append(solution.VarName)
        for i in range(len(nonzero_x)):
            print('\n x = %s' %nonzero_x[i])
        model._logfile.write('\n**** New solution # %d, Obj %g, Current Gap %g%%, Best Obj %g, Best Gap %g%%, Elapsed time %g,'
               % (solcnt, obj, current_gap, objbst, best_gap, model.cbGet(GRB.Callback.RUNTIME)))
        for i in range(len(nonzero_x)):
            model._logfile.write('\n x = %s' %nonzero_x[i])

    elif where == GRB.Callback.MIPNODE: #Currently exploring a MIP node
        # MIP node callback
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(model._vars) #Values from the node relaxation solution at the current node
            model.cbSetSolution(model.getVars(), x)
    elif where == GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)

    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        model._logfile.write(msg)
        
def run_file_parallel(args: dict):

    clustering_label = args['clustering_label']
    result = args['clustering_result']
    label = args['catalog_label']
    too_close_threshold = args['too_close_threshold']
    
    location = '/home/idies/workspace/Storage/tunguyen52/Nway/Data/log/' + filename

    ## Get a particular island
    candidate_list, df, catalog_list, Sigma_list = get_candidate_list(result, label, clustering_label)
    if len(np.unique(catalog_list)) <= 1:
        print(f"These sources from island {clustering_label} are orphans because they come from the same catalog")
        logfile = open(os.path.join(location,'Island_'+str(clustering_label)+'.log'), 'w')
        logfile.write(f"These sources from island {clustering_label} are orphans because they come from the same catalog\n")
        logfile.write("\n".join(str(item) for item in candidate_list))
        logfile.close()
        return 0
    
    ## Find the total number of unique catalog in the island
    catalog_test_list = []
    for catalog_source_pair in candidate_list:
        if catalog_source_pair[1] not in catalog_test_list:
            catalog_test_list.append(catalog_source_pair[1])
    catalog_total = len(catalog_test_list)
    
    ## Total number of hypothesized objects
    detection_total = max(Counter(catalog_list).values()) * 2
 
    ## Find kappa values 
    kappa_dict = {} 
    kappa_rad_dict = {}
    ln_kappa_rad_dict = {}

    for i in range(len(candidate_list)):
        kappa_dict[(candidate_list[i][0], candidate_list[i][1])] = 1/(Sigma_list[i])**2 #in arcsecond 
        kappa_rad_dict[(candidate_list[i][0], candidate_list[i][1])] = 1/((Sigma_list[i]*np.pi/180/3600)**2) #in radian
        ln_kappa_rad_dict[(candidate_list[i][0], candidate_list[i][1])] = ln(1/((Sigma_list[i]*np.pi/180/3600)**2))

    sigma_min = min(Sigma_list)
    sigma_max = max(Sigma_list)
    sigma = sigma_max
    
    ## Compute pairwise distance

    distance_dict = compute_distance_dictionary(candidate_list, df)
    new_distance_dict = {}
    
    ## Too close heuristic constraint
    catalog_constraint_list = []
    for pair,distance in distance_dict.items():
        if distance < too_close_threshold*sigma:
            new_distance_dict[pair]=distance
    sorted_distance_dict = {k: v for k, v in sorted(new_distance_dict.items(), key=lambda item: item[1])}
    
    # Make a chain of sources that are close to each other (up to some threshold)
    
    iset = set([frozenset(s) for s in list(sorted_distance_dict.keys())])  # Convert to a set of sets
    filter_result = []
    while(iset):                  # While there are sets left to process:
        nset = set(iset.pop())      # Pop a new set
        check = len(iset)           # Does iset contain more sets
        while check:                # Until no more sets to check:
            check = False
            for s in iset.copy():       # For each other set:
                if nset.intersection(s):  # if they intersect:
                    check = True            # Must recheck previous sets
                    iset.remove(s)          # Remove it from remaining sets
                    nset.update(s)          # Add it to the current set
        filter_result.append(tuple(nset))  # Convert back to a list of tuples
    # Sort this chain based on the number of sources in the chain
    filter_result_dict = {}
    for item in filter_result:
        filter_result_dict[item] = len(item)
    filter_result_dict = {k: v for k, v in sorted(filter_result_dict.items(), key=lambda item: item[1], reverse=True)}

    ######################### ILP Formulation ###########################

    mo = Model("likelihood")

    M = np.ceil(sum_of_distance(candidate_list, df)/(4*sigma**2))
    var_x_dict = {}
    var_y_dict = {}
    var_z_dict = {}
    var_w_dict = {}
    t_list = []
    p_list = []
    a_list = [0]

    ########################### SET VARIABLES ###########################

    # Compute a_list
    for catalog_index in range(catalog_total - 1):
        a_list.append(ln(catalog_index + 2) - ln(catalog_index + 1))

    # Variables for x
    for subset_index in range(detection_total):
        for catalog_source_pair in candidate_list:
            var_x_dict[('x', subset_index, catalog_source_pair)] = mo.addVar(vtype=GRB.BINARY, name=str(('x', subset_index, catalog_source_pair)))

    # Variables for y
    for subset_index in range(detection_total):
        for product_of_catalog_source_pairs in list(itertools.combinations(candidate_list, r = 2)):
            if product_of_catalog_source_pairs[0][1] != product_of_catalog_source_pairs[1][1]:
                var_y_dict[('y', subset_index, product_of_catalog_source_pairs[0], product_of_catalog_source_pairs[1])] = mo.addVar(vtype=GRB.BINARY, name=str(('y', subset_index, product_of_catalog_source_pairs[0],product_of_catalog_source_pairs[1])))

    # Variables for z
    for subset_index in range(detection_total):
        for catalog_index in range(catalog_total + 1):
            var_z_dict[('z', subset_index, catalog_index)] = mo.addVar(vtype=GRB.BINARY, name=str(('z', subset_index, catalog_index)))

    # Variables for t
    for subset_index in range(detection_total):  
        t_list.append(mo.addVar(lb = 0, vtype=GRB.CONTINUOUS, name=str(('t', subset_index))))

    # Variables for p
    for subset_index in range(detection_total):  
        p_list.append(mo.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name=str(('p', subset_index))))

    # Variables for w
    for subset_index in range(detection_total):
        for catalog_index in range(catalog_total):
            var_w_dict[('w', subset_index, catalog_index)] = mo.addVar(vtype=GRB.BINARY, name=str(('w', subset_index, catalog_index)))

    ########################### SET OBJECTIVES ###########################

    # Set objective
    mo.setObjective(quicksum(p_list[subset_index] + quicksum(var_w_dict[('w', subset_index, catalog_index)]*a_list[catalog_index] for catalog_index in range(catalog_total)) + t_list[subset_index] for subset_index in range(detection_total)), GRB.MINIMIZE)

    ########################### SET CONSTRAINTS ###########################

    # All detections (i,c) needs to belong to some subset (S_j)
    # Equation A3
    for catalog_source_pair in candidate_list:    
        x_constraint = []
        for variable in var_x_dict.keys():
            if variable[-1] == catalog_source_pair:
                x_constraint.append(var_x_dict[variable])
        mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.EQUAL, rhs = 1)

    # Every subset takes no more than 1 detection from each catalog
    # Equation A4
    for subset_index in range(detection_total):
        for catalog_index in range(catalog_total):
            x_constraint = []
            for variable in var_x_dict.keys():
                if (variable[1] == subset_index) & (variable[-1][1] == catalog_index):
                    x_constraint.append(var_x_dict[variable])
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.LESS_EQUAL, rhs = 1)

    # Definition of variables y
    # Equations A5 - A7
    for subset_index in range(detection_total):
        for product_of_catalog_source_pairs in list(itertools.combinations(candidate_list, r = 2)):
            if product_of_catalog_source_pairs[0][1] != product_of_catalog_source_pairs[1][1]:
                mo.addConstr(var_y_dict[('y', subset_index, product_of_catalog_source_pairs[0], product_of_catalog_source_pairs[1])] >= var_x_dict[('x', subset_index, product_of_catalog_source_pairs[0])] + var_x_dict[('x', subset_index, product_of_catalog_source_pairs[1])] - 1)
                mo.addConstr(var_y_dict[('y', subset_index, product_of_catalog_source_pairs[0], product_of_catalog_source_pairs[1])] <= var_x_dict[('x', subset_index, product_of_catalog_source_pairs[0])])
                mo.addConstr(var_y_dict[('y', subset_index, product_of_catalog_source_pairs[0], product_of_catalog_source_pairs[1])] <= var_x_dict[('x', subset_index, product_of_catalog_source_pairs[1])])

    # The cardinality of any subset from a partition P is from 0 to K
    # Equation A8
    for subset_index in range(detection_total):
        z_constraint = []
        for catalog_index in range(catalog_total + 1):
            z_constraint.append(var_z_dict[('z', subset_index, catalog_index)])
        mo.addConstr(lhs = quicksum(variable for variable in z_constraint), sense=GRB.EQUAL, rhs = 1)

    # Definition of variables w
    # Equation A9
    for subset_index in range(detection_total):
        w_constraint = []
        x_constraint = []
        for catalog_index in range(catalog_total):
            w_constraint.append(var_w_dict[('w', subset_index, catalog_index)])
        for catalog_source_pair in candidate_list:    
            x_constraint.append(var_x_dict[('x', subset_index, catalog_source_pair)])
        mo.addConstr(lhs = quicksum(variable for variable in w_constraint), sense=GRB.EQUAL, rhs = quicksum(variable for variable in x_constraint))
        for w_index in range(len(w_constraint) - 1):
            mo.addConstr(w_constraint[w_index] >= w_constraint[w_index + 1])

    # Definition of variables t
    # Equation A10
    for subset_index in range(detection_total):
        for catalog_index in range(1, catalog_total + 1):
            mo.addConstr(lhs = t_list[subset_index], sense=GRB.GREATER_EQUAL, 
                         rhs = -M * (1 - var_z_dict[('z', subset_index, catalog_index)]) + quicksum(1/(4*catalog_index * sigma**2)*var_y_dict[('y', subset_index, product_of_catalog_source_pairs[0], product_of_catalog_source_pairs[1])]*get_distance(product_of_catalog_source_pairs, df) for product_of_catalog_source_pairs in list(itertools.combinations(candidate_list, r = 2)) if product_of_catalog_source_pairs[0][1] != product_of_catalog_source_pairs[1][1]))

    # Definition of variables p
    # Equation A13
    for subset_index in range(detection_total):
        mo.addConstr(lhs = p_list[subset_index], sense=GRB.GREATER_EQUAL, 
                         rhs = ln(2*1/(sigma*np.pi/180/3600)**2)*(1 - quicksum(var_w_dict[('w', subset_index, catalog_index)] for catalog_index in range(catalog_total))) - ln(2*1/(sigma*np.pi/180/3600)**2)*var_z_dict[('z', subset_index, 0)])

    # Definition of variables z
    # Equation A11 - 12
    for subset_index in range(detection_total):
        x_constraint = []
        for catalog_source_pair in candidate_list:
            x_constraint.append(var_x_dict[('x', subset_index, catalog_source_pair)])
        for catalog_index in range(catalog_total + 1):
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.LESS_EQUAL,
                         rhs = catalog_index * var_z_dict[('z', subset_index, catalog_index)] + catalog_total * (1 - var_z_dict[('z', subset_index, catalog_index)]))
            mo.addConstr(lhs = quicksum(variable for variable in x_constraint), sense=GRB.GREATER_EQUAL,
                         rhs = catalog_index * var_z_dict[('z', subset_index, catalog_index)])

    # Heuristic constraints:
    #Too far to belong to the same object
    distance_dict = compute_distance_dictionary(candidate_list, df)
    too_far_constr=[]
    for pair,distance in distance_dict.items():
        if distance > 8*sigma:
            for subset_index in range(detection_total):
                too_far_constr.append(var_y_dict[tuple(['y']+[subset_index]+list(pair))])
    for variable in too_far_constr:
        variable.ub = 0
    
    # Too close constraint
    if len(filter_result) == 1:
        for source in filter_result[0]:
            var_x_dict[('x', 0, source)].lb = 1
        for pair in list(itertools.combinations(filter_result[0], r = 2)):
            if pair[0][1] > pair[1][1]:
                pair = list(pair)
                pair.reverse()
                var_y_dict[tuple(['y']+[0]+pair)].lb = 1
            else:
                var_y_dict[tuple(['y']+[0]+list(pair))].lb = 1
    if len(filter_result) > 1:
        for index in range(2):
            for source in list(filter_result_dict.keys())[index]:
                var_x_dict[('x', index, source)].lb = 1
            for pair in list(itertools.combinations(list(filter_result_dict.keys())[index], r = 2)):
                if pair[0][1] > pair[1][1]:
                    pair = list(pair)
                    pair.reverse()
                    var_y_dict[tuple(['y']+[index]+pair)].lb = 1
                else:
                    var_y_dict[tuple(['y']+[index]+list(pair))].lb = 1
    mo.update()
    logfile = open(os.path.join(location,'Island_'+str(clustering_label)+'.log'), 'w')
    # Pass data into my callback function

    mo._logfile = logfile
    mo._vars = mo.getVars()

    # Solve model
    mo.Params.MIPGap = 0.005
    mo.optimize(mycallback)
    logfile.close()
    #print(time.time() - start_1)
    
with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    cmat, X, result, label = pickle.load(f)
    

num_processors = 16
args = [
{
    'clustering_label': clustering_label
    , 'clustering_result': result
    , 'catalog_label': label
    , 'too_close_threshold': 0.1
}
for clustering_label in np.unique(result.labels_)[:1000] if clustering_label != -1
]

p=Pool(processes = num_processors)
output = p.map(run_file_parallel, args)
