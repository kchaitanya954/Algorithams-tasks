# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:45:02 2020

@author: Admin
"""
#import required libraries
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import timeit
from decimal import Decimal
 
#constant function
def cons(v):
    '''returns a constant value C'''
    return "C"

#sum of elements function
def soe(v):
    ''' function which return sum of elements in an array'''
    k=0
    for i in range(len(v)):
        k=k+v[i]
    return k

#product of elements function
def poe(v):
    '''returns the product of elements of the list'''
    k=1
    for i in range(len(v)):
        k=k*v[i]
    return k

#polynomial function
def poly(v):
    '''value of polynomial at x=1.5'''
    for i in range(len(v)):
        k=Decimal(v[i])*(Decimal(1.5)**Decimal(i))
    return k

#honors function
def hon(v):
    '''value of polynomial at x=1.5'''
    k=0
    for i in range(len(v)):
        k=Decimal(v[i])+Decimal(1.5)*Decimal(k)
    return k

#bubble sort of elements of v
def bubsort(v):
    for i in range(len(v)-1):
        for j in range(0, len(v)-1-i):
            if v[j+1]<v[j]:
                temp=v[j]
                v[j]=v[j+1]
                v[j+1]=temp
    return v

#quicksort for elements of v
def quicksort(v):
    if len(v)<=1:
        return v
    else: 
        p, v=v[-1], v[:-1]
    g=[]
    l=[]
    for i in v:
        if i>p:
            g.append(i)
        else:
            l.append(i)
    return (quicksort(l)+[p]+quicksort(g))
#defining insertion for timsort
def InsertionSort(array):

    for x in range (1, len(array)):
        for i in range(x, 0, -1):
            if array[i] < array[i - 1]:
                t = array[i]
                array[i] = array[i - 1]
                array[i - 1] = t
            else:
                break
            i = i - 1
    return array
#defining merge for timsort
def Merge(aArr, bArr):
    
    a = 0
    b = 0
    cArr = []

    while a < len(aArr) and b < len(bArr):
        if aArr[a] <= bArr[b]:
            cArr.append(aArr[a])
            a = a + 1
        elif aArr[a] > bArr[b]:
            cArr.append(bArr[b])
            b = b + 1
       
    while a < len(aArr):
        cArr.append(aArr[a])
        a = a + 1

    while b < len(bArr):
        cArr.append(bArr[b])
        b = b + 1

    return cArr 

#tim sort
def timsort(v):

    for x in range(0, len(v), 64):
        v[x : x + 64] = InsertionSort(v[x : x + 64])
    RUNinc = 64
    while RUNinc < len(v):
        for x in range(0, len(v), 2 * RUNinc):
            v[x : x + 2 * RUNinc] = Merge(v[x : x + RUNinc], v[x + RUNinc: x + 2 * RUNinc])
        RUNinc = RUNinc * 2
    return v

#matrix multiplication
def matmul(a,b):
    for i in range(len(a)):
        for j in range(len(b)):
            c=np.zeros((len(a),len(a)), float)
            
            for k in range(len(a)):
                c[i,j]=c[i,j]+a[i,k]*b[k,j]
    return c

#Fitting function
def func(x, a, b):
    return a*x**b

# Initial guess for the parameters
initialGuess = [1.0,1.0]    
#x values for the fitted function
xFit = np.arange(0.0, 2000, 0.01)
#Experimental size of array and execution time data points    
n = np.arange(2000)

t1=[]
t2=[]
t3=[]
t4=[]
t5=[]
t6=[]
t7=[]
t8=[]
t9=[]
for i in range(1,2001):
    #v=np.random.rand(i)
    a=np.random.rand(i,i)
    b=np.random.rand(i,i)
    # t1.append(timeit.timeit('cons(v)','from __main__ import cons, v', number=5)/5)
    # t2.append(timeit.timeit('soe(v)','from __main__ import soe, v', number=5)/5)
    # t3.append(timeit.timeit('poe(v)','from __main__ import poe, v', number=5)/5)
    # t4.append(timeit.timeit('poly(v)','from __main__ import poly, v', number=5)/5)
    # t5.append(timeit.timeit('hon(v)','from __main__ import hon, v', number=5)/5)
    # t6.append(timeit.timeit('bubsort(v)','from __main__ import bubsort, v', number=5)/5)
    # t7.append(timeit.timeit('quicksort(v)','from __main__ import quicksort, v', number=5)/5)
    # t8.append(timeit.timeit('timsort(v)','from __main__ import timsort, v', number=5)/5)
    t9.append(timeit.timeit('matmul(a,b)','from __main__ import matmul, a,b', number=5)/5)

#Plot experimental data points of constant function
plt.plot(n, t1, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t1, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of constant function')
plt.show() 

 
#Plot experimental data points of sum of elements function
plt.plot(n, t2, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t2, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of sum of elements function')
plt.show() 

#Plot experimental data points of product of elements function
plt.plot(n, t3, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t3, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of product of elements function')
plt.show() 

#Plot experimental data points of polynomial
plt.plot(n, t4, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t4, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of polynomial')
plt.show() 

#Plot experimental data points of polynomial by honors method
plt.plot(n, t5, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t5, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of polynomial by honors method')
plt.show() 

#Plot experimental data points of bubble sorting
plt.plot(n, t6, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t6, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of bubblesort')
plt.show() 

#Plot experimental data points of quick sorting
plt.plot(n, t7, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t7, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of quicksort')
plt.show() 

#Plot experimental data points of tim sorting
plt.plot(n, t8, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t8, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of timsort')
plt.show() 

#Plot experimental data points for matrix multiplication
plt.plot(n, t9, 'bo')

#Perform the curve-fit
popt, pcov = curve_fit(func, n, t9, initialGuess)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r')
 
plt.xlabel('Size of an array')
plt.ylabel('Execution time')
plt.title('Execution time of matrix multiplication')
plt.show() 
