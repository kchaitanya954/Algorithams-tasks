#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:45:58 2020

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


g = {'a':{'b':10,'c':3},'b':{'c':1,'d':2},'c':{'b':4,'d':8,'e':2},'d':{'e':7},'e':{'d':9}}

# dijikstra method for shortest path
def D(graph, start, end):
    shortest_distance = {}
    visited = {}
    unvisited = graph
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
        unvisited.pop(minNode)
    return shortest_distance