#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:28:50 2020

@author: chaitanya
"""
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import random

# generating noise data
xk=np.arange(0, 3.003, 0.003)
fk=1/(xk**2-3*xk+2)

yk=np.zeros(1001)
for i in range(1001):
    if fk[i]< -100:
        yk[i]=-100+np.random.normal(0,1)
    elif fk[i]> 100:
        yk[i]=100+np.random.normal(0,1)
    else:
        yk[i]=fk[i]+np.random.normal(0,1)

# defining rational approximatin function.
def fun(x):
    fun=(x[0]*xk+x[1])/(xk**2+x[2]*xk+x[3])
    return fun

# defining least squares of rational approximation function
def D(x):
    return sum((fun(x)-yk)**2)

# defining residual function
def residual(x):
    return fun(x)-yk

#defining jacobian matrix for rational approximation function
def jacobian(x):
    j=np.empty((xk.size, x.size))
    den=(xk**2+x[2]*xk+x[3])
    j[:,0]=xk/den
    j[:,1]=1/den
    j[:,2]=-xk*fun(x)/den
    j[:,3]=-fun(x)/den
    return j


# intial approximations
x0=np.array([0.5,0.5,1,1])
# caluclating Rational approximation using Nelder mead.
res_nm=optimize.minimize(D, x0, method='Nelder-Mead',\
                      options={'maxiter':1000, 'disp':True, 'fatol':0.001} )

# caluclating rational approximation using evenberg-Marquardt algorithm
res_lm=optimize.least_squares(residual, x0, jac=jacobian,\
                               method='lm',ftol=0.001 )   

# caluclating rational apprximation using simulated anneling.
lw = [-10] * 4
up = [10] * 4
bounds=list(zip(lw, up))

res_sm = optimize.dual_annealing(D, bounds,maxiter=4, accept=1, \
                                  seed=1234 )

# caluclating rational approximation using differential evalution.
res_de=optimize.differential_evolution(D, bounds, maxiter=1000, atol=0.001)

# Caluclating rational approximation using particle swarn optimizaion
bounds = [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]  # upper and lower bounds of variables
nv = 4  # number of variables
mm = 1  # if minimization problem, mm = -1; if maximization problem, mm = 1
 
# PARAMETERS OF PSO
particle_size = 120  # number of particles
iterations = 200  # max number of iterations
w = 0.8  # inertia constant
c1 = 1  # cognative constant
c2 = 2  # social constant
initial_fitness = float("inf")  # for minimization problem
 

# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position
 
        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity
 
    def evaluate(self, objective_function):
        self.fitness_particle_position = objective_function(self.particle_position)
        if self.fitness_particle_position < self.fitness_local_best_particle_position:
            self.local_best_particle_position = self.particle_position  # update the local best
            self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
     
    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()
 
            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity
 
    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]
 
            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]
 
def PSO( objective_function, bounds, particle_size, iterations):
    fitness_global_best_particle_position = initial_fitness
    global_best_particle_position = []
    swarm_particle = []
    for i in range(particle_size):
        swarm_particle.append(Particle(bounds))
    A = np.array([1])
    k=0
    for i in range(iterations):
        for j in range(particle_size):
            swarm_particle[j].evaluate(objective_function) 
            if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                global_best_particle_position = list(swarm_particle[j].particle_position)
                fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
            
        for j in range(particle_size):
            swarm_particle[j].update_velocity(global_best_particle_position)
            swarm_particle[j].update_position(bounds)
 
        A=np.append(A,fitness_global_best_particle_position)
        k+=1
        if all(abs(np.diff(A))>0.001):
            continue
        else:
            break
    return k, global_best_particle_position, fitness_global_best_particle_position/2
    
res_pso=PSO(D, bounds, particle_size, iterations)



y_nm=fun(res_nm.x)
y_lm=fun(res_lm.x)
y_sm=fun(res_sm.x)
y_de=fun(res_de.x)
y_pso=fun(res_pso[1])

plt.plot(xk, yk,'bo' ,label='Noisy data')
plt.plot(xk, y_pso, '-c', label='Particle swarm')
plt.plot(xk, y_sm, '-g', label='Simulated annealing')
plt.plot(xk, y_lm, '-r', label='Levenberg-Marquardt')
plt.plot(xk, y_nm, '-y' , label='Nelder-mead')
plt.plot(xk, y_de, '-k', label='Didderential Evolution')
plt.title('Rational approximation of noisy data')
plt.ylabel('Noisy data')
plt.xlabel('x-axis')
plt.legend()
plt.show()


