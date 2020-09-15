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
for i in range(1,2001):
    v=np.random.rand(i)
    t1.append(timeit.timeit('cons(v)','from __main__ import cons, v', number=5)/5)
    t2.append(timeit.timeit('soe(v)','from __main__ import soe, v', number=5)/5)
    t3.append(timeit.timeit('poe(v)','from __main__ import poe, v', number=5)/5)
    t4.append(timeit.timeit('poly(v)','from __main__ import poly, v', number=5)/5)
    t5.append(timeit.timeit('hon(v)','from __main__ import hon, v', number=5)/5)

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
