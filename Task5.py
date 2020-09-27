#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:01:07 2020

@author: chaitanya
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 


# genarating a random adjacency matrix for a undirected and unweighted graph
def a_matrix(x):
    ''' generates a adjacancy matrix of size x'''
    g=np.zeros([x+1, x+1])
    g[:, 0]= np.arange(x+1)
    g[0, :]=np.arange(x+1)
    for i in range(1,x+1):
        for j in range(1,x+1):
            if i !=j:
                g[i, j]=g[j, i]=np.random.choice(np.arange(0,2),p=[0.9,0.1] )
                if sum(sum(g[1:, 1:]))>=400:
                    break        
        if sum(sum(g[1:,1:]))>=400:
                break  
    return g


adj_mat=a_matrix(100)
print(adj_mat)
# generating adjacancy list from the matrix
def a_list(m):
    '''generates a adjancy list from the matrix'''
    a={}
    for i in range(1,len(m)):
        l=np.array([])
        for j in range(1, len(m)):             
            if m[i][j]==1:
                l=np.append(l,m[0][j])
        # if len(l)!=0:
            a[i]=l
    return a
adj_list=a_list(adj_mat)

G = nx.from_numpy_matrix(np.array(adj_mat[1:, 1:]))  
nx.draw(G, with_labels=True)
plt.title('Graph of random adjacency matrix of 100 vertices and 200 edges')
plt.show()

# generating graph from the adjacancy matrix 
def graph(m):
    G=nx.Graph()
    for i in range(1,len(m)):
        for j in range(1, len(m)):
            if adj_mat[i][j]==1:
                G.add_edge(i, j)
                G.add_edge(j,i)
    return G

# visulizing the graph
G=graph(adj_mat)
nx.draw(G, with_labels=True)
plt.title('Graph of random adjacency matrix of connected components')
plt.show()

# returns the connected components.
def connected_components_list(graph):
       visited = []
       connected_components = []
       for node in graph.nodes:
           if node not in visited:
               cc = [] #connected component
               visited, cc = dfs(graph, node, visited, cc)
               connected_components.append(cc)
       return connected_components

# the dfs algorith to check the visited vertices
def dfs(graph, start, visited, path):
    if start in visited:
        return visited, path
    visited.append(start)
    path.append(start)
    for node in graph.neighbors(start):
        visited, path = dfs(graph, node, visited, path)
    return visited, path
Graph = nx.from_numpy_matrix(np.array(adj_mat[1:, 1:]))  

# prints the number of connected components and connections
connections=connected_components_list(Graph)
print(len(connections))
for i in connections:
    print(i)
    
# Shortest path between any two vertices
def BFS_SP(graph, start, goal): 
    explored = []       
    queue = [[start]]       
    # If the desired node is reached 
    if start == goal: 
        print("Same Node") 
        return
      
    # Loop to traverse the graph with the help of the queue 
    while queue: 
        path = queue.pop(0) 
        node = path[-1] 
          
        # Codition to check if the current node is not visited 
        if node not in explored: 
            neighbours = graph[node] 
              
            # Loop to iterate over the neighbours of the node 
            for neighbour in neighbours: 
                new_path = list(path) 
                new_path.append(neighbour) 
                queue.append(new_path) 
                  
                # Condition to check if the neighbour node is the goal 
                if neighbour == goal: 
                    print("Shortest path = ", new_path) 
                    return new_path, len(new_path)-1
            explored.append(node)   
    # Condition when the nodes are not connected 
    
    return print("Connecting path doesn't exist ") 

