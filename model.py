import numpy as np
import math
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import os
import random
import pickle
import copy
import matplotlib.pyplot as plt
import networkx as nx



#Network Models


def label_shuffling_comm(network, prop):
    # Shuffles node labels _while_ maintaining community structure
    N = network.number_of_nodes()
    num_comms = len({network.nodes[x]['comm'] for x in range(0, N)})
    for comm in range(1, num_comms+1):
        nodes_in_comm = [x for x in network.nodes() if network.nodes[x]['comm'] == comm]
        frac_N = int(prop*len(nodes_in_comm))
        labels = random.sample(nodes_in_comm, frac_N)
        shuffled_labels = labels[:]
        np.random.shuffle(shuffled_labels)
        # Create shuffle mapping
        # (x<->y)
        mapping_one = {labels[c] : N+shuffled_labels[c] for c in range(frac_N)} # (x->z)
        mapping_two = {shuffled_labels[c]: labels[c] for c in range(frac_N)} # (y->x)
        mapping_three = {N+shuffled_labels[c] : shuffled_labels[c] for c in range(frac_N)} # (z->y)
        # Apply shuffling
        if comm == 1:
            shuffled_network = nx.relabel_nodes(G = network, mapping = mapping_one)
        else:
            shuffled_network = nx.relabel_nodes(G = shuffled_network, mapping = mapping_one)
        shuffled_network = nx.relabel_nodes(G = shuffled_network, mapping = mapping_two)
        shuffled_network = nx.relabel_nodes(G = shuffled_network, mapping = mapping_three)
    return shuffled_network

def label_shuffling_no_comm(network, prop):
    # Shuffles node labels _without_ maintaining community structure

    # Get correct number of labels to shuffle
    N = network.number_of_nodes()
    frac_N = int(N*prop) # Alter to be proper rounding later
    # Grab shuffled labels
    labels = random.sample(range(N), frac_N)
    shuffled_labels = labels[:]
    np.random.shuffle(shuffled_labels)
    # Create shuffle mapping
    # (x<->y)
    mapping_one = {labels[c] : N+shuffled_labels[c] for c in range(frac_N)} # (x->z)
    mapping_two = {shuffled_labels[c]: labels[c] for c in range(frac_N)} # (y->x)
    mapping_three = {N+shuffled_labels[c] : shuffled_labels[c] for c in range(frac_N)} # (z->y)
    # Apply shuffling
    shuffled_network = nx.relabel_nodes(G = network, mapping = mapping_one)
    shuffled_network = nx.relabel_nodes(G = shuffled_network, mapping = mapping_two)
    shuffled_network = nx.relabel_nodes(G = shuffled_network, mapping = mapping_three)
    return shuffled_network
#

def M_SIR(G,b1,b2,mu,seed,time_steps):
    g1,g2=G[0],G[1]
    S,I,R=[],[],[]
    state_dict={i:'S' for i in g1.nodes()}
    state_dict[seed]='I'
    for _ in range(time_steps):
        infections,recoveries=[],[]
        for i in state_dict:
            if state_dict[i]=='I':
                for j in g1.neighbors(i):
                    if state_dict[j]=='S' and np.random.random()<b1:
                        infections.append(j)
                for j in g2.neighbors(i):
                    if state_dict[j]=='S' and np.random.random()<b2:
                        infections.append(j)
                if np.random.random()<mu:
                    recoveries.append(i)
        for node in infections:
            state_dict[node]='I'
        for node in recoveries:
            state_dict[node]='R'
        S.append(list(state_dict.values()).count('S'))
        I.append(list(state_dict.values()).count('I'))
        R.append(list(state_dict.values()).count('R'))
    return np.array(S),np.array(I),np.array(R)



"""
    Algorithm for creating communities
"""
def MSBM_COMM(N,C,div,mu1,mu2,k1,k2,p_s,p_o=1):
    #ng_time = time.perf_counter()
    #l1,l2 -> # of communities in each layer
    # l1 -> families
    # l2 -> classes
    l1 = int(C) 
    l2 = int(C/div)
    g1,g2 = {},{}

    # list contaning community ids for each node 
    # index -> node ID 
    # value -> community ID
    l1_comms = []
    for i in range(0,l1):
        for j in range(0,int(N/l1)):
            l1_comms.append(i)

    l2_comms = [l1_comms[i]%l2 for i in range(N)]
    #new average degree after removed nodes in layer 2 (ie, the bigger layer)
    # check this value
    
    node_list=list(range(N))
    edges_l1,edges_l2=[],[]

    #get community assignments for nodes in both the layers
    # dictionary specifiying neighboring community nodes for each node
    # key: community ID 
    # value: list of nodes in that nodes community
    l1_comm_node_list = {}
    l2_comm_node_list = {}

    
    
    # Perform shuffling based on p_s
    # FR - use binomial distribution with p_s to shuffle!
    # relabel second layer, thereotically simpler
    nodes_to_be_shuffled = int(p_s*N)
    for i in range(0,nodes_to_be_shuffled,2): 
        switch_one = random.randint(0,N-1)
        switch_two = random.randint(0,N-1)
        l2comm_one = l2_comms[switch_one]
        l2comm_two = l2_comms[switch_two]
        l2_comms[switch_one] = l2comm_two
        l2_comms[switch_two] = l2comm_one
    
    # FR - recreate dictionary of communities!
    #remove nodes based on overlap
    removed_nodes = set(random.sample(list(range(N)), int((1-p_o)*N)))
    #new average degree after removed nodes in layer 2 (ie, the bigger layer)
    #if p_o < 1: 
    #    new_k2 = int(p_o*k2)
    #else:
    #    new_k2 = k2
    new_k2 = k2

    for i in range(0,N):
        l1_comm_node_list[l1_comms[i]] = []
        g1[i] = set()
        g2[i] = set()
        l2_comm_node_list[l2_comms[i]] = []
    for i in range(0,N):
        l1_comm_node_list[l1_comms[i]].append(i)
        if i not in removed_nodes:
            l2_comm_node_list[l2_comms[i]].append(i)
    N_l2 = int(N*(p_o))
    # q -> # of nodes in each community
    q1 = int(N/l1)
    q2 = int(N_l2/l2)
    pin_l1,pout_l1=k1*(1-mu1)/(q1-1),mu1*k1/(N-q1)
    pin_l2,pout_l2=new_k2*(1-mu2)/(q2-1),mu2*new_k2/(N_l2-q2)
    if pin_l1 > 1 or pin_l2 > 1 or pout_l1 > 1 or pout_l2 > 1:
        print("Linking probabilities greater than 1 - restructure parameters!")
        return

    c1 = 0
    c2 = 0
    # loop through pairs of communities to assign links between pairs 
    #layer one loop
    # FR - write equation out
    #in_edge_combos = math.comb(q1,2)
    # factorial calculation done here manually
    in_edge_combos = (q1*(q1-1))/(2)
    out_edge_combos = q1*q1
    sbm_time = time.perf_counter()
    occupied_edges = set()
    # FR - use dictionary nodelist
    # L1 group assignment
    for c1 in range(0,l1):
        in_edges = np.random.binomial(in_edge_combos,pin_l1)
        e = 0
        #allocate edges to communities 
        while e < in_edges:
            a_in = (random.choice(l1_comm_node_list[c1]))
            b_in = (random.choice(l1_comm_node_list[c1]))
            if a_in != b_in and (a_in,b_in) not in occupied_edges:
                occupied_edges.add((a_in,b_in))
                g1[a_in].add(b_in)
                g1[b_in].add(a_in)
                e += 1
        for c2 in range(c1+1,l2):
            out_edges = np.random.binomial(out_edge_combos,pout_l1)
            e = 0
            while e < out_edges:
                a_out = (random.choice(l1_comm_node_list[c1]))
                b_out = (random.choice(l1_comm_node_list[c2]))
                if (a_out,b_out) not in occupied_edges:
                    occupied_edges.add((a_out,b_out))
                    g1[a_out].add(b_out)
                    g1[b_out].add(a_out)
                    e += 1
    # second layer assignment
    in_edge_combos = (q2*(q2-1))/(2)
    out_edge_combos = q2*q2
    sbm_elapsed = time.perf_counter() - sbm_time
    sbm_time = time.perf_counter()
    occupied_edges = set()
    for c1 in range(0,l2):
        in_edges = np.random.binomial(in_edge_combos,pin_l2)
        e = 0
        #randomly choose nodes in community 
         # c1,c2 -> communities to link
         # looping within a community
        while e < in_edges:
            a_in = (random.choice(l2_comm_node_list[c1]))
            b_in = (random.choice(l2_comm_node_list[c1]))
            if a_in != b_in and (a_in,b_in) not in occupied_edges:
                occupied_edges.add((a_in,b_in))
                g2[a_in].add(b_in)
                g2[b_in].add(a_in)
                e += 1
         # connecting edges between communities 
        for c2 in range(c1+1,l2):
            out_edges = np.random.binomial(out_edge_combos,pout_l2)
            e = 0
            while e < out_edges:
                a_out = (random.choice(l2_comm_node_list[c1]))
                b_out = (random.choice(l2_comm_node_list[c2]))
                if (a_out,b_out) not in edges_l2:
                    edges_l2.append((a_out,b_out))
                    g2[a_out].add(b_out)
                    g2[b_out].add(a_out)
                    e += 1
    # assign inner edges
    NMIscore = normalized_mutual_info_score(l1_comms,l2_comms)
    return [[g1,g2],NMIscore,[l1_comm_node_list,l2_comm_node_list]]


def SBM_trilayer(C,N,ps1,ps2,ks,mus):
    k1,k2,k3=ks[0],ks[1],ks[2]
    mu1,mu2,mu3=mus[0],mus[1],mus[2]
    l1_comms,l2_comms,l3_comms = [],[],[]
    for i in range(0,C):
        for _ in range(0,int(N/C)):
            l1_comms.append(i)
            l2_comms.append(i)
            l3_comms.append(i)
    


    nodes_to_be_shuffled = int(ps1*N)
    for i in range(0,nodes_to_be_shuffled,2): 
        switch_one = random.randint(0,N-1)
        switch_two = random.randint(0,N-1)
        l2comm_one = l2_comms[switch_one]
        l2comm_two = l2_comms[switch_two]
        l2_comms[switch_one] = l2comm_two
        l2_comms[switch_two] = l2comm_one

    
    nodes_to_be_shuffled = int(ps2*N)
    for i in range(0,nodes_to_be_shuffled,2): 
        switch_one = random.randint(0,N-1)
        switch_two = random.randint(0,N-1)
        l3comm_one = l3_comms[switch_one]
        l3comm_two = l3_comms[switch_two]
        l3_comms[switch_one] = l3comm_two
        l3_comms[switch_two] = l3comm_one

    lil1=[[] for _ in range(C)]
    lil2=[[] for _ in range(C)]
    lil3=[[] for _ in range(C)]

    for n in range(N):
        lil1[l1_comms[n]].append(n)
        lil2[l2_comms[n]].append(n)
        lil3[l3_comms[n]].append(n)
    


    g1,g2,g3={i:set() for i in range(N)},{i:set() for i in range(N)},{i:set() for i in range(N)}



### Create Layer 1

    occupied_edges=set()
    for com_num,com in enumerate(lil1):
        cs=len(com)
        if cs>1:
            pinl1,poutl1=k1*(1-mu1)/(cs-1),mu1*k1/(N-cs)
            if pinl1>1:
                pinl1=0.9999
            in_edges = np.random.binomial(int(cs*(cs-1)/2),pinl1)
            e = 0
            #randomly choose nodes in community looping within a community
            while e < in_edges:
                a_in = (random.choice(com))
                b_in = (random.choice(com))
                if a_in != b_in and (a_in,b_in) not in occupied_edges:
                    occupied_edges.add((a_in,b_in))
                    g1[a_in].add(b_in)
                    g1[b_in].add(a_in)
                    e += 1

        # connecting edges between communities 
        for com2 in lil1[com_num:]:
            occupied_edges=set()
            cs_2=len(com2)
            out_edges = np.random.binomial(int(cs*cs_2),poutl1)
            e = 0
            while e < out_edges:
                a_out = (random.choice(com))
                b_out = (random.choice(com2))
                if (a_out != b_out) and (a_out,b_out) not in occupied_edges:
                    occupied_edges.add((a_out,b_out))
                    g1[a_out].add(b_out)
                    g1[b_out].add(a_out)
                    e += 1
            
    
    ### Create Layer 2

    occupied_edges=set()
    for com_num,com in enumerate(lil2):
        cs=len(com)
        
        if cs>1:
            pinl2,poutl2=k2*(1-mu2)/(cs-1),mu2*k2/(N-cs)
            if pinl2>1:
                pinl2=0.9999
            in_edges = np.random.binomial(int(cs*(cs-1)/2),pinl2)
            e = 0
            #randomly choose nodes in community looping within a community
            while e < in_edges:
                a_in = (random.choice(com))
                b_in = (random.choice(com))
                if a_in != b_in and (a_in,b_in) not in occupied_edges:
                    occupied_edges.add((a_in,b_in))
                    g2[a_in].add(b_in)
                    g2[b_in].add(a_in)
                    e += 1

        # connecting edges between communities 
        for com2 in lil2[com_num:]:
            occupied_edges=set()
            cs_2=len(com2)
            out_edges = np.random.binomial(int(cs*cs_2),poutl2)
            e = 0
            while e < out_edges:
                a_out = (random.choice(com))
                b_out = (random.choice(com2))
                if (a_out != b_out) and (a_out,b_out) not in occupied_edges:
                    occupied_edges.add((a_out,b_out))
                    g2[a_out].add(b_out)
                    g2[b_out].add(a_out)
                    e += 1
    

    ### Create Layer 3

    occupied_edges=set()
    for com_num,com in enumerate(lil3):
        cs=len(com)
        
        if cs>1:
            pinl3,poutl3=k3*(1-mu3)/(cs-1),mu3*k3/(N-cs)
            if pinl2>1:
                pinl2=0.9999
            in_edges = np.random.binomial(int(cs*(cs-1)/2),pinl3)
            e = 0
            #randomly choose nodes in community looping within a community
            while e < in_edges:
                a_in = (random.choice(com))
                b_in = (random.choice(com))
                if a_in != b_in and (a_in,b_in) not in occupied_edges:
                    occupied_edges.add((a_in,b_in))
                    g3[a_in].add(b_in)
                    g3[b_in].add(a_in)
                    e += 1

        # connecting edges between communities 
        for com3 in lil3[com_num:]:
            occupied_edges=set()
            cs_3=len(com3)
            out_edges = np.random.binomial(int(cs*cs_3),poutl3)
            e = 0
            while e < out_edges:
                a_out = (random.choice(com))
                b_out = (random.choice(com3))
                if (a_out != b_out) and (a_out,b_out) not in occupied_edges:
                    occupied_edges.add((a_out,b_out))
                    g2[a_out].add(b_out)
                    g2[b_out].add(a_out)
                    e += 1
    nmi2,nmi3=normalized_mutual_info_score(l1_comms,l2_comms),normalized_mutual_info_score(l1_comms,l3_comms)

    return g1,g2,g3,[nmi2,nmi3]



"""
Generates two layers of a stochastic block model (SBM) based on input data.

Parameters:
c1 (dict): A dictionary representing the node-to-community assignments in layer 1.
            Keys are node IDs, and values are community IDs.
c2 (dict): A dictionary representing the node-to-community assignments in layer 2.
            Keys are node IDs, and values are community IDs.
k1 (float): Average degree parameter for layer 1.
k2 (float): Average degree parameter for layer 2.
mu1 (float): Mixing parameter for within-community edges in layer 1.
                Controls the probability of connecting nodes within the same community.
mu2 (float): Mixing parameter for within-community edges in layer 2.
                Controls the probability of connecting nodes within the same community.

Returns:
g1 (dict): A dictionary representing the adjacency list of layer 1.
            Keys are node IDs, and values are sets of neighboring node IDs.
            Represents the generated graph for layer 1.
g2 (dict): A dictionary representing the adjacency list of layer 2.
            Keys are node IDs, and values are sets of neighboring node IDs.
            Represents the generated graph for layer 2.
"""

def SBM_data_based(c1,c2,k1,k2,mu1,mu2):
    N=len(c1)
    lil1=[[] for _ in range(len(set(c1.values())))]
    lil2=[[] for _ in range(len(set(c2.values())))]
    
    g1,g2={i:set() for i in c1},{i:set() for i in c2}


    for i in c1:
        lil1[c1[i]].append(i)
    
    for i in c2:
        lil2[c2[i]].append(i)
### Create Layer 1

    occupied_edges=set()
    for com_num,com in enumerate(lil1):
        cs=len(com)
        if cs>1:
            pinl1,poutl1=k1*(1-mu1)/(cs-1),mu1*k1/(N-cs)
            if pinl1>1:
                pinl1=0.9999
            in_edges = np.random.binomial(int(cs*(cs-1)/2),pinl1)
            e = 0
            #randomly choose nodes in community looping within a community
            while e < in_edges:
                a_in = (random.choice(com))
                b_in = (random.choice(com))
                if a_in != b_in and (a_in,b_in) not in occupied_edges:
                    occupied_edges.add((a_in,b_in))
                    g1[a_in].add(b_in)
                    g1[b_in].add(a_in)
                    e += 1

        # connecting edges between communities 
        for com2 in lil1[com_num:]:
            occupied_edges=set()
            cs_2=len(com2)
            out_edges = np.random.binomial(int(cs*cs_2),poutl1)
            e = 0
            while e < out_edges:
                a_out = (random.choice(com))
                b_out = (random.choice(com2))
                if (a_out != b_out) and (a_out,b_out) not in occupied_edges:
                    occupied_edges.add((a_out,b_out))
                    g1[a_out].add(b_out)
                    g1[b_out].add(a_out)
                    e += 1
            
    
    ### Create Layer 2

    occupied_edges=set()
    for com_num,com in enumerate(lil2):
        cs=len(com)
        
        if cs>1:
            pinl2,poutl2=k2*(1-mu2)/(cs-1),mu2*k2/(N-cs)
            if pinl2>1:
                pinl2=0.9999
            in_edges = np.random.binomial(int(cs*(cs-1)/2),pinl2)
            e = 0
            #randomly choose nodes in community looping within a community
            while e < in_edges:
                a_in = (random.choice(com))
                b_in = (random.choice(com))
                if a_in != b_in and (a_in,b_in) not in occupied_edges:
                    occupied_edges.add((a_in,b_in))
                    g2[a_in].add(b_in)
                    g2[b_in].add(a_in)
                    e += 1

        # connecting edges between communities 
        for com2 in lil2[com_num:]:
            occupied_edges=set()
            cs_2=len(com2)
            out_edges = np.random.binomial(int(cs*cs_2),poutl2)
            e = 0
            while e < out_edges:
                a_out = (random.choice(com))
                b_out = (random.choice(com2))
                if (a_out != b_out) and (a_out,b_out) not in occupied_edges:
                    occupied_edges.add((a_out,b_out))
                    g2[a_out].add(b_out)
                    g2[b_out].add(a_out)
                    e += 1

    return g1,g2
    


def LFR(n,t1,t2,mu,avg_k,max_k):
    N,Mu,T1,T2,maxk,k=str(n),str(mu),str(t1),str(t2),str(max_k),str(avg_k)
    s='./benchmark -N '+N+' -mu '+Mu+ ' -maxk ' +maxk  + ' -k '+k  +' -t1 ' +T1+' -t2 ' +T2
    os.system(s)
    x=np.loadtxt('community.dat')
    coms={int(x[i][0])-1:int(x[i][1])-1 for i in range(len(x))}
    return coms


def MSBM_heterogeneous_communities(p_s=0,N=10000,t2=10,mu1=0.025,mu2=0.025,k1=6,k2=6):
    c1=LFR(N,10,t2,0.5,11,10000)
    c2=copy.copy(c1)
    nodes_to_be_shuffled = int(p_s*N)
    for _ in range(0,nodes_to_be_shuffled,2): 
        switch_one,switch_two = random.randint(0,N-1),random.randint(0,N-1)
        l2comm_one,l2comm_two = c2[switch_one],c2[switch_two]
        c2[switch_one],c2[switch_two] = l2comm_two,l2comm_one
    g1,g2=SBM_data_based(c1,c2,k1=k1,k2=k2,mu1=mu1,mu2=mu2)
    nmi=normalized_mutual_info_score(list(c1.values()),list(c2.values()))
    return [g1,g2],nmi


def LFR_het_deg(n=10000,t1=2,mu=0.025,average_k=4,max_k=10,community=100):
    #function to generate LFR network as a networkx object and obtain community assignments
    # Cast parameters as strings
    N = str(n)
    Mu = str(mu)
    T1 = str(t1)
    maxk = str(max_k)
    average_k=str(average_k)
    min_community = str(community)
    # Concatenate LFR binary shell command as a string
    # s='./benchmark -N '+N+' -mu '+Mu+ ' -maxk ' +maxk  + ' -k '+k  +' -t1 ' +T1+' -t2 ' +T2  +' -minc ' +min_community
    s='./benchmark -N '+N+' -mu '+Mu+ ' -k '+average_k+' -maxk ' +maxk  +' -t1 ' +T1+  ' -minc ' +min_community +' -maxc ' +min_community
    # >>> LFR generation >>>
    # Call LFR binary and generate networks
    os.system(s)
    # Format resultant network as networkx Graph
    x=np.loadtxt('network.dat')
    edges=[(int(x[i][0])-1,int(x[i][1])-1) for i in range(len(x))]
    g=nx.Graph(edges)
    # Format resultant node partition as dict
    x=np.loadtxt('community.dat')
    coms={int(x[i][0])-1:int(x[i][1]) for i in range(len(x))}

    return g, coms



def MSBM_COMM_het_deg(N=10000,t1=2,corr=False,q=10):
    #code to create multiplex SBM with q equal communities and heterogeneous degree distributions 
    #create the networks
    #the communities are chosen randomly, i.e., the resulting networks are completely uncorrelated
    g1,c1=LFR_het_deg(mu=0.025,t1=t1)
    g2,c2=LFR_het_deg(mu=0.025,t1=t1)
    #nodes per community
    C=int(N/q)
    #to create networks with correlated communities
    if corr:
    #relabel the communities based on their community assignment
        relabel1={}
        #count community assignments for each community
        cn=[0]*(q+1)
        for i in c1:
            relabel1[i]=(c1[i]-1)*(C)+cn[c1[i]]
            cn[c1[i]]+=1

        relabel2={}
        #count community assignments for each community
        cn=[0]*(q+1)
        #community assignments
        for i in c2:
            relabel2[i]=(c2[i]-1)*(C)+cn[c2[i]]
            cn[c2[i]]+=1
        #relabel appropriately

        g1=nx.relabel_nodes(g1,relabel1)
        g2=nx.relabel_nodes(g2,relabel2)
    #convert to edge list format
    G1={i:set(g1.neighbors(i)) for i in g1.nodes()}
    G2={i:set(g2.neighbors(i)) for i in g2.nodes()}
    return [G1,G2]
