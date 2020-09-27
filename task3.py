# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:21:19 2020

@author: Admin
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# generating noisy data
alpha=np.random.rand()
beta=np.random.rand()
delta=[]
for i in range(101):
    delta.append(np.random.normal(0,1))
xk=np.arange(0, 1.01, 0.01)
yk=alpha*xk+beta+delta

# defining square error for linear approximation 
def DL(x):
    err=0
    a=x[0]
    b=x[1]
    #xk, yk=args
    for i in range(101):
        err+=((a*xk[i]+b) -yk[i])**2
    return err

# defining square error for rational approximation 
def DR(x):
    err = 0
    a=x[0]
    b=x[1]
    for i in range(101):
        err += ((a/ (xk[i] * b+1) )- yk[i]) ** 2
    return err

# gradient for linear square errors function
def gradDL(x):
    a=x[0]
    b=x[1]
    y=a*xk+b
    ga=2*sum(xk*(y-yk))
    gb=2*sum(y-yk)
    return np.asarray((ga, gb))

# gradient for rational square error function
def gradDR(x):
    a=x[0]
    b=x[1]
    y=a/(1+b*xk)
    ga=2*sum((y-yk )*(y/a))
    gb=2*sum((yk-y)*y*(xk*y/a))
    return np.asarray((ga, gb))

# hesian matrix for linear square errors function
def Hessian_L(x):
    a=x[0]
    b=x[1]
    y=a*xk+b
    gaa=2*sum(xk**2)
    gab=2*sum(xk)
    gba=2*sum(xk)
    gbb=2*101
    return np.asarray([[gaa, gab], [gba, gbb]])

# hesian matrix for rational square errors function
def Hessian_R(x):
    a=x[0]
    b=x[1]
    y=a/(xk*b+1)
    gaa=2*sum((y/a)**2)
    gab=2*sum((xk)*(-2*a+yk+b*xk*yk)/(1+b*xk)**3)
    gba=2*sum(((xk)*(b*xk*yk+yk-2*a))/(a+b*xk)**3)
    gbb=2*sum((2*a*(xk**2))*(-2*b*xk*yk-2*yk+3*a)/(1+b*xk)**4)
    return np.asarray([[gaa, gab], [gba, gbb]])

#defining linear approximation function
def FL(x):
    return x[0]*xk+x[1]

#defining rational approximation function
def FR(x):
    return x[0]/(1+x[1]*xk)

#defining residuals for linear approximation function
def residuel_L(x):
    return FL(x)-yk
#defining residuals for rational approximation function
def residuel_R(x):
    return FR(x)-yk

#defining jacobian matrix for linear approximation function
def jacobian_L(x):
    j=np.zeros((xk.size, x.size))
    j[:,0]=xk
    j[:,1]=1
    return j

#defining jacobian matrix for rational approximation function
def jacobian_R(x):
    j=np.empty((xk.size, x.size))
    j[:,0]=1/(1+x[1]*xk)
    j[:,1]=-FR(x)*xk/x[0]
    return j


# linear approximation using gradient descent
# al,bl=0.5,0.5
# gamma=0.00001
# err=0.001
# pal, pbl=1,1
# i=0
# while pal>err and pbl>err:
#     y=al*xk+bl
#     a0l=al
#     b0l=bl
#     gal= gradDL([al,bl])[0]
#     gbl= gradDL([al,bl])[1]
#     al=al-gamma*gal
#     bl=bl-gamma*gbl
#     # precision=np.sqrt((al-a0)**2+(bl-b0)**2)
#     pal= abs(al-a0l)
#     pbl=abs(bl-b0l)
#     i+=1
# yl_g=FL([al,bl])

def gradient_descent_l(x,y):
    m_curr = b_curr = 1
    iterations = 1000
    learning_rate = 0.00001
  
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = sum([val**2 for val in (y-y_predicted)])
        md = -(2)*sum(x*(y-y_predicted))
        bd = -(2)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
    x_est=np.array([m_curr, b_curr])
    return x_est

gl=gradient_descent_l(xk,yk)
yl_g=FL(gl)
# Rational approximation using gradient descent
# ar, br=0.5,0.5
# gamma=0.00001
# err=0.001
# # precision=1
# par, pbr=1,1
# j=0
# while par>err and pbr> err:
#     y=ar/(1+br*xk)
#     a0r=ar
#     b0r=br
#     ga=gradDR([ar,br])[0]
#     gb=gradDR([ar,br])[1]
#     ar=ar-gamma*ga
#     br=br-gamma*gb
#     par= abs(ar-a0r)
#     pbr=abs(br-b0r)
#     j+=1
# yr_g=FR([ar,br])

def gradient_descent_r(x,y):
    m_curr = b_curr = 1
    iterations = 1000
    learning_rate = 0.00001

    for j in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = sum([val**2 for val in (y-y_predicted)])
        md = -(2)*sum((y-y_predicted)*(y_predicted/m_curr))
        bd = -(2)*sum((y-y_predicted)*(x*y_predicted**2)/m_curr)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
       
    x_est=np.array([m_curr, b_curr])
    return x_est

gr=gradient_descent_r(xk,yk)
yr_g=FR(gr)

#initial approximation
x0=np.array([0.5,0.5])

#caluclating the linear approximation function using conjugate method
l_conjugate_gradient=optimize.fmin_cg(DL, x0, fprime=gradDL, gtol=0.001 )
yl_cg=FL(l_conjugate_gradient)

#caluclating the rational approximation function using conjugate gradient method    
r_conjugate_gradient=optimize.fmin_cg(DR, x0, fprime=gradDR, gtol=0.001)
yr_cg=FR(r_conjugate_gradient)

#caluclating the linear approximation function using Newton's method
l_newton= optimize.minimize(DL, x0, jac=gradDL, method='Newton-CG', \
                               hess=Hessian_L, tol=0.001,options={'disp':True})
yl_n=FL(l_newton.x)

#caluclating the rational approximation function using Newton's method
r_newton= optimize.minimize(DR, x0, jac=gradDR,  method='Newton-CG',\
                               hess=Hessian_R, tol=0.001,options={'disp':True}) 
yr_n=FR(r_newton.x)

#caluclating the linear approximation function using Levenberg-Marquardt                         
res_l=optimize.least_squares(residuel_L, x0, jac=jacobian_L,\
                             method='lm',ftol=0.001 )
yl_lm=FL(res_l.x)
#caluclating the Rational approximation function using Levenberg-Marquardt
res_R=optimize.least_squares(residuel_R, x0, jac=jacobian_R,\
                               method='lm',ftol=0.001 )
yr_lm=FR(res_R.x)      

# plotting linear approximation of noisy data
plt.scatter(xk, yk)
plt.plot(xk, yl_g, '-y', label='Gradient descent')
plt.plot(xk, yl_cg, '-r', label='Conjugate gradient')
plt.plot(xk, yl_n, '-g', label='Newton method')
plt.plot(xk, yl_lm, '-k', label='Levenberg-Marquardt')
plt.xlabel('x-axis')
plt.ylabel('Noisy data')
plt.title('Linear approximation of noisy data')
plt.legend()
plt.show()

# plotting rational approximation of noisy data
plt.scatter(xk, yk)
plt.plot(xk, yr_g, '-y', label='Gradient descent')
plt.plot(xk, yr_cg, '-r', label='Conjugate gradient')
plt.plot(xk, yr_n, '-g', label='Newton method')
plt.plot(xk, yr_lm, '-k', label='Levenberg-Marquardt')
plt.xlabel('x-axis')
plt.ylabel('Noisy data')
plt.title('Rational approximation of noisy data')
plt.legend()
plt.show()