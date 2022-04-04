# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:09:18 2022

@author: fahad mostafa @ TTU

"""


import numpy as np

from numpy import linalg 

#===== Proximal_Gradiet method  ============# 
 
kmax = 10000
tol = 1e-12
rho = 1e-6 # various choice
def Problem2(x,A,b):        # cost function
    y = 0.5* np.matmul((np.matmul(A,x)-b).T,(np.matmul(A,x)-b)) + rho*0.5* np.matmul(x.T,x)
    return float(y)

# prox operator for ridge 
def prox_operator_ridge_regression(y, L, rho):
    v = y/(rho/L+1)
    return v
    
def proximal_grad_ridge(A,b,x,t,rho):
    y = np.zeros((4,1))
    L = linalg.norm((np.matmul(A.T,A))) 
    diff = 2*tol
    cost_current = Problem2(x,A,b)
    print('Initial cost value', cost_current)
    k = 1
    while (diff > tol and k < kmax):
        xold = x.copy()
        y = x- 1./L*((np.matmul(A.T,np.matmul(A,x)-b))) 
        x = prox_operator_ridge_regression(y, L, rho)
 #       cost_old = cost_current
 #       cost_current =  Problem2(x, A, b)
        k = k + 1
 #       diff = abs(cost_old - cost_current)
        diff = np.linalg.norm(x-xold,2)
    return x , k
    
    

if __name__=='__main__':
    A = np.array([[1.,2,3],[4,5,6],[7,8,9]])
    x = np.array([[0.],[0],[0]])
    b = np.array([[1.],[1],[1]])
    
# =============================================================================
    t = 0.5
    x, k= proximal_grad_ridge(A,b,x,t,rho)
    print('optimal solution:',x,k)
    print('optimal function value:', Problem2(x,A,b))

# =============================================================================