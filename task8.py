#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:55:11 2020

@author: chaitanya
"""

import numpy as np
import matplotlib.pyplot as plt
import string
import random
import timeit
from scipy.optimize import curve_fit
from scipy.linalg import lu


# Naive string matching algorithm
def nsm(T, P):
    n=len(T)
    m=len(P)
    s=[]
    for i in range(n-m+1):
        if P==T[i:i+m]:
            s.append(i)
    return s

# Rabin Karp algorithm
def rkm(T, P, d, q):
    n=len(T)
    m=len(P)
    h=(d**(m-1)) % q
    p=0
    t=0
    result=[]
    for i in range(m):
        p=(d*p+ord(P[i])) % q
        t=(d*t+ ord(T[i])) % q
        
    for s in range(n-m+1):
        if p==t:
            match=True
            for i in range(m):     
                if P!=T[s:s+m]:
                    match=False
                    break
            if match:
                result=result+[s]
        if s<n-m:
            t=(t-h*ord(T[s]))%q
            t=(t*d+ord(T[s+m]))%q
            t=(t+q)%q
    return result

NO_OF_CHARS = 256

# generates the nest state for finite automata  
def getNextState(P, M, state, x): 
    ''' 
    calculate the next state  
    '''
    if state < M and x == ord(P[state]): 
        return state+1
  
    i=0

    for ns in range(state,0,-1): 
        if ord(P[ns-1]) == x: 
            while(i<ns-1): 
                if P[i] != P[state-ns+1+i]: 
                    break
                i+=1
            if i == ns-1: 
                return ns  
    return 0

# generates the transistion table
def computeTF(P, M): 
    ''' 
    This function builds the TF table which  
    represents Finite Automata for a given pattern 
    '''
    global NO_OF_CHARS 
  
    TF = [[0 for i in range(NO_OF_CHARS)] for _ in range(M+1)] 
  
    for state in range(M+1): 
        for x in range(NO_OF_CHARS): 
            z = getNextState(P, M, state, x) 
            TF[state][x] = z 
  
    return TF 
 
# finite automata for string matching
def search(P, T): 
    ''' 
    Prints all occurrences of pat in txt 
    '''
    global NO_OF_CHARS 
    M = len(P) 
    N = len(T) 
    TF = computeTF(P, M)     
  
    # Process txt over FA. 
    state=0
    for i in range(N): 
        state = TF[state][ord(T[i])] 
        if state == M: 
            print("Pattern found at index: {}". format(i-M+1)) 
  

# txt = "AABAACAADAABAAABAA"
# pat = "AABA"
# search(P, T) 

# Knuth-Morris-Pratt algorithm
def kmp_match(T, P):
    """
    Implementation of the Knuth-Morris-Pratt (KMP) algorithm in Python.
    This algorithm finds valid shifts to locate subsequence `subseq` in
    sequence `seq`.
    """
    n = len(T)
    m = len(P)
    pi = prefix(P)
    k = 0
    s=[]
    for i in range(n):
        while k > 0 and P[k] != T[i]:
            k = pi[k-1]
        if P[k] == T[i]:
            k += 1
        if k == m:
            s.append((i+1)-m)
            k = 0
    return s

# caluclates the prefix array for kmp algorithm
def prefix(P):
    """Checks seq for valid shifts against itself."""
    m = len(P)
    pi = np.arange(m)
    k = 0

    for i in pi[1:]:
        while k > 0 and P[k+1] != P[i]:
            k = pi[k]
        if P[k+1] == P[i]:
            k = k + 1
        pi[i] = k

    return pi

# geberates a randon string of any given size
def s(size, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


# lup decomposition
def lup(mat): 
    n=len(mat)
    lower = np.zeros([n,n]); 
    upper = np.zeros([n,n]); 
                  
    # Decomposing matrix into Upper  
    # and Lower triangular matrix 
    for i in range(n): 
  
        for k in range(i, n):  
  
            sum = 0; 
            for j in range(i): 
                sum += (lower[i][j] * upper[j][k]); 
  
            upper[i][k] = mat[i][k] - sum; 
  
        for k in range(i, n): 
            if (i == k): 
                lower[i][i] = 1
            else: 
  
                sum = 0; 
                for j in range(i): 
                    sum += (lower[k][j] * upper[j][i]); 
  
                    lower[k][i] = int((mat[k][i] - sum) /
                                       upper[i][i]); 
  
    return lower, upper
    
# LUP decomposition method to solve linear equations
def LUP_solve(A, b):
    A=A.copy()
    n=len(A)
    # b=b.transpose()
    x=np.zeros(n)
    y=np.zeros(n)
    P=np.array(lu(A)[0])
    L=np.array(lu(A)[1])
    U=np.array(lu(A)[2])
    pi=np.zeros(n)
    for i in range(n):
        pi[i]=np.where(P[i]==1)[0][0]
    for i in range(n):
        y[i]=b[int(pi[i])]-sum(L[i][:i]*y[:i])
    
    for i in range(n-1, -1, -1):
        x[i]=(y[i]-sum(U[i][i+1:]*x[i+1:]))/U[i][i]
        
    return x

# Matrix inverse using LUP decomposition
def LUP_inv(A):
    A=A.copy()
    n=len(A)
    Ainv=np.zeros(n**2).reshape([n,n])
    b=np.identity(n)
    for i in range(n):
        Ainv[i]=LUP_solve(A, b[i])
    return Ainv

# least square fit using matrices
def lsf(x,y, n):
    m=len(x)
    A=np.zeros([m,n])
    for i in range(m):
        A[i]=np.array([x[i]**i for i in range(n)])
    
    k=np.dot(A.transpose(), A)
    A_pinv=np.dot(LUP_inv(k), A.transpose())
    C=np.dot(A_pinv, y)
    return C

# regression function
def func(x, a, b):
    return a*(x**(b))

# caluclating time stamps for tme complexity
t_lup=[]
t_solve=[]
t_inv=[]
t_lsf=[]
for i in range(1, 100):
    A=np.random.rand(i**2).reshape([i,i])
    b=np.random.rand(i)
    x=np.random.rand(i)
    y=x**2
    k=i
    t_lup.append(timeit.timeit('lup(A)','from __main__ import lup,\
                                A', number=5)/5)
    
    t_solve.append(timeit.timeit('LUP_solve(A,b)','from __main__ import LUP_solve,\
                                A, b', number=5)/5)

    t_inv.append(timeit.timeit('LUP_inv(A)','from __main__ import LUP_inv,\
                                A', number=5)/5)
                               
    t_lsf.append(timeit.timeit('lsf(x, y, k)','from __main__ import lsf,\
                               x, y, k', number=5)/5)

# initial guess for regression line
initialGuess=[0.1,0.1]
n=np.arange(1,100)
# perforem curve fit
popt, pcov = curve_fit(func, n, t_solve, initialGuess)

#Plot the fitted function
plt.plot(n, func(n, *popt), '-r', label='Expected time')
plt.plot(n, t_solve, label='Execution time')
plt.legend()
plt.title('Execution time for solving system of linear equations LUP decomposition')
plt.xlabel('Size of Matrix')
plt.ylabel('Execution time')
plt.show()

# perforem curve fit

popt, pcov = curve_fit(func, n, t_inv, initialGuess)

#Plot the fitted function
plt.plot(n, func(n, *popt), '-r', label='Expected time')
plt.plot(n, t_inv, label='Execution time')
plt.legend()
plt.title('Execution time for matrix inverse')
plt.xlabel('Size of Matrix')
plt.ylabel('Execution time')
plt.show()

# perforem curve fit

popt, pcov = curve_fit(func, n, t_lsf, initialGuess)

#Plot the fitted function
plt.plot(n, func(n, *popt), '-r', label='Expected time')
plt.plot(n, t_lsf, label='Execution time')
plt.legend()
plt.title('Execution time for least square fit using matrices')
plt.xlabel('Size of Matrix')
plt.ylabel('Execution time')
plt.show()