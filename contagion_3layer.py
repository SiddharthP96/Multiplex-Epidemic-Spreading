
from tkinter import W
import networkx as nx
import numpy as np
import random


# Gillespie with three layers
def SIR_3G(g1,g2,g3,b1,b2,b3,mu,state_vector,active_one,active_two,active_three,infected,s,r):
    
    #layer recovery probability (same as l2 since multiplex)
    p1 = random.random()
    p2 = random.random()
    # p3 = random.random()

    prop_li1 = b1 * len(active_one)
    prop_li2 = b2 * len(active_two)
    prop_li3 = b3 * len(active_three)
    prop_r = mu*len(infected)
    
    norm = prop_li1 + prop_li2 + prop_li3 + prop_r 
    #event control flow - either infection or recovery occurs
    #recovery occurence

    #Spread event on layer one 
    if p2 <= (prop_li1 / norm) and len(active_one)>0:
    #remove nx
        l = random.choice(active_one)
        n = l[1]
        state_vector[n] = "I"
        s -= 1
        infected.append(n)
        neigh = g1[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_one.append((n, z))
            if state_vector[z] == "I":
                active_one.remove((z, n))
        neigh = g2[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_two.append((n, z))
            if state_vector[z] == "I":
                active_two.remove((z, n))
        neigh = g3[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_three.append((n, z))
            if state_vector[z] == "I":
                active_three.remove((z, n))
    
    #Spreading event on layer two
    if (p2 <= (prop_li1 + prop_li2) / norm) and (p2 > (prop_li1 / norm)) and len(active_two)>0: 
        l = random.choice(active_two)
        n = l[1]        
        state_vector[n] = "I"
        s -= 1
        infected.append(n)
        neigh = g1[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_one.append((n, z))
            if state_vector[z] == "I":
                active_one.remove((z, n))
        neigh = g2[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_two.append((n, z))
            if state_vector[z] == "I":
                active_two.remove((z, n))
        neigh = g3[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_three.append((n, z))
            if state_vector[z] == "I":
                active_three.remove((z, n))

    #Spreading event on layer three
    if (p2 <= (prop_li1 + prop_li2 + prop_li3) / norm) and (p2 >= ((prop_li1+prop_li2)/norm)) and len(active_three)>0: 
        l = random.choice(active_three)
        n = l[1]        
        state_vector[n] = "I"
        s -= 1
        infected.append(n)
        neigh = g1[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_one.append((n, z))
            if state_vector[z] == "I":
                active_one.remove((z, n))
        neigh = g2[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_two.append((n, z))
            if state_vector[z] == "I":
                active_two.remove((z, n))
        neigh = g3[n]
        for z in neigh:
            if state_vector[z] == "S":
                active_three.append((n, z))
            if state_vector[z] == "I":
                active_three.remove((z, n))
    
    
    #Recovery event, agnostic of layer
    if p2 > ((prop_li1 + prop_li2 + prop_li3) / norm): 
        n = random.choice(infected)
        infected.remove(n)
        state_vector[n] = "R"
        r += 1
        neigh_one = g1[n]
        for z in neigh_one:
            if state_vector[z] == "S":
                active_one.remove((n, z))
        neigh_two = g2[n]
        for z in neigh_two:
            if state_vector[z] == "S":
                active_two.remove((n, z))
        neigh_three = g3[n]
        for z in neigh_three:
            if state_vector[z] == "S":
                active_three.remove((n, z))

    return s,len(infected),r,(1.0 / (norm) * np.log(1.0 / p1))
#@g1,g2 - dictionaries, keys: node-ids, values: neighbors



def gillespie_sir_3sim(g1,g2,g3,b1,b2,b3,mu,ft,seed,p_i):
    #initializing all dictionaries
    ft = 1000000
    mu=1
    state_vector = {}
    #stores percentage of S,I,R
    s,i,r = [],[],[]
    time_sim = []
    active_one,active_two,active_three=[],[],[]
    # number of susceptibles
    state_vector,i_keys,r_keys,n_s= set_3state(g1,g2,g3,seed,p_i)
    #seed neighbors and create active graphs
    for n in i_keys: 
        neigh_one = g1[n]
        neigh_two = g2[n] 
        neigh_three = g3[n] 
        for k in neigh_one: 
            if state_vector[k] == "S": 
                active_one.append((n,k))
        for k in neigh_two: 
            if state_vector[k] == "S":
                active_two.append((n,k))
        for k in neigh_three: 
            if state_vector[k] == "S":
                active_three.append((n,k))
    #Now, begin time loop
    t = 0.0 
    i.append(len(i_keys)/len(state_vector))
    time_sim.append(t)
    r.append(len(r_keys)/len(state_vector))
    #check below line
    s.append(len(i_keys)-1/len((state_vector)))
    n_i = len(i_keys)
    n_r = len(r_keys)
    while n_i > 0:
        #run one step of gillespie
        n_s,n_i,n_r,dt = SIR_3G(g1,g2,g3,b1,b2,b3,mu,state_vector,active_one,active_two,active_three,i_keys,n_s,n_r)
        #increment time
        t = t + dt
        #returning percentage valus of different compartments
        s.append(n_s/len(state_vector))
        i.append(n_i/len(state_vector))
        r.append(1-(n_s+n_i)/len(state_vector))
        time_sim.append(t)
    return time_sim,np.array(s),np.array(i),np.array(r)

# sets states of all nodes in matrix
def set_3state(g1,g2,g3,seed,p_i):
    state_vector = {}
    s_num = 0

    nodes=list(set(g1.keys()).union(set(g2.keys())))
    nodes = list(set(nodes).union(set(g3.keys())))
    nodes.remove(seed)
    immune=random.sample(nodes,k=int(len(nodes)*p_i))
    state_vector[seed] = "I"
    #setting state vector
    for node in nodes:
        if node in immune:
            state_vector[node] = "R"
        else:
            state_vector[node] = "S"
            s_num+=1

    return state_vector,[seed],immune,s_num
