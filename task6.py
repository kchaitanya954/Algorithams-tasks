#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:47:26 2020

@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
import timeit


def w_matrix(x):
    g=np.zeros([x+1, x+1])
    g[:, 0]= np.arange(x+1)
    g[0, :]=np.arange(x+1)
    for i in range(1,x+1):
        for j in range(1,x+1):
            if i !=j:
                g[i, j]=g[j, i]=np.random.choice(np.arange(0,10),\
                                p=[0.82, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,0.02, 0.02, 0.02] )
                if sum(sum(g[1:,1:]==0))<=9000:
                    break
        if sum(sum(g[1:,1:]==0))<=9000:
            break
    return g

adj_mat=w_matrix(100)

G = nx.from_numpy_matrix(np.array(adj_mat[1:,1:]))  
nx.draw(G, with_labels=True)
plt.title('Graph of random adjacency matrix of 100 vertices and 200 edges')
plt.show()

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
Graph = nx.from_numpy_matrix(np.array(adj_mat))  

# prints the number of connected components and connections
connections=connected_components_list(Graph)
for i in connections:
    print(i)
    
    
def a_list(m):
    '''generates a adjancy list from the matrix'''
    a={}
    for i in range(1,len(m)):
        l={}
        for j in range(1, len(m)):             
            if m[i][j]!=0:
                l[m[0][j]]=m[i][j]

            a[i]=l
    return a
adj_list=a_list(adj_mat)

def Dijkstra(graph, start, end):
    shortest_distance = {}
    visited = {}
    unvisited = [i for i in graph]
    path = []
    for node in unvisited:
        shortest_distance[node] = float('inf')
    shortest_distance[start] = 0

    while unvisited:
        minNode = None
        for node in unvisited:
            if minNode is None:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node

        for childNode, weight in graph[minNode].items():
            if weight + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = weight + shortest_distance[minNode]
                visited[childNode] = minNode
        unvisited.remove(minNode)
    currentNode = end
    while currentNode != start:
        path.insert(0,currentNode)
        currentNode = visited[currentNode]
    return shortest_distance[end], path

# bellman ford mothod for shortest path
g = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'b':4,'d':8,'e':2},'d':{'e':7},'e':{'d':9}}

def Bellman_ford(graph, start, end):
    shortest_distance={}
    visited ={}
    unvisited=graph.copy()
    for node in unvisited:
        shortest_distance[node]=float('inf')
    shortest_distance[start]=0
    path=[]
    for i in range(len(adj_list)-1):
        for u in graph:
            for v in graph[u]:
                if shortest_distance[v]>shortest_distance[u]+graph[u][v]:
                    shortest_distance[v]=shortest_distance[u]+graph[u][v]
                    visited[v]=u
    for u in graph:
        for v in graph[u]:
            assert shortest_distance[v] <= \
                shortest_distance[u] + graph[u][v], "Negative cycle exists"
    currentNode = end
    while currentNode != start:
        path.insert(0,currentNode)
        currentNode = visited[currentNode]                
    return shortest_distance[end], path

# initialize the values
graph= adj_list
start=16
end=95
# execution time for Dijkstra method for average of 10 runs
t_Dijkstra=timeit.timeit('Dijkstra(graph, start, end)','from __main__ import Dijkstra,\
                  graph, start, end', number=10)/10
                  
# execution time for Bellman_ford method for average of 10 runs
t_Bellman_ford=timeit.timeit('Bellman_ford(graph, start, end)','from __main__ import Bellman_ford,\
                  graph, start, end', number=10)/10
                  
                  
# generating 10x10 grid with 30 obstacles
def grid(x):
    ''' 1 indcates the blocks, 0 indicates the path'''
    g=np.zeros([x, x])
    for i in range(x):
        for j in range(x):
            g[i, j]=np.random.choice(np.arange(0,2),p=[0.65,0.35] )
            if sum(sum(g))>=30:
                break        
        if sum(sum(g))>=30:
                break  
    return g

g=grid(10)
# A start method to find shortest path between two points.
class Node():
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.position == other.position   
    
def astar(maze, start, end):
    start_node = Node(None, start)
    # h is heuristic distance
    # f is total distance
    # g is diatance from start
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    open_list = []
    closed_list = []
    open_list.append(start_node)
    while len(open_list) > 0:
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        open_list.pop(current_index)
        closed_list.append(current_node)
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or\
                node_position[1] > (len(maze[len(maze) - 1]) - 1) or node_position[1] < 0:
                continue
            if maze[node_position[0]][node_position[1]] != 0:
                continue
            new_node = Node(current_node, node_position)
            children.append(new_node)
        for child in children:
            for closed_child in closed_list:
                if child == closed_child:
                    continue
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                    (child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue
            open_list.append(child)


