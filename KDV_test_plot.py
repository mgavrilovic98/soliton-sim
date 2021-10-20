#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:21:49 2021

@author: teddywest
"""
#creating the kdv function
##imports and built-in fucntions
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, cosh
from scipy.fftpack import diff as diff
from scipy.integrate import odeint
## creating the kdv equation 
def sech_squared(x):
    return (1.0/cosh(x))**2

def kdv_exact(x, A):
    w = sqrt(2/A)
    return A * sech_squared(x/w)

def diss_kdv(u, t, L, mu):
    ux   = diff(u, period=L)
    uxxx = diff(u, period=L, order=3)
    uxx  = diff(u, period=L, order=2)
    dudt = -6*u*ux - uxxx + mu*uxx    
    return dudt

def diss_kdv_solution(u0, t, L, mu):
    solu = odeint(diss_kdv, u0, t, args=(L,mu), mxstep=4096)
    return solu

def diss_waveshape(x,*p):   
    return p[0]*sech_squared( (x-p[1])/p[2] )



## setting conditions for test kdv equation
L = 50.0
N = 64
dx = L / (N - 1.0)
x = np.linspace(0, (1-1.0/N)*L, N)

## got these constants from an approximation 
u0 = kdv_exact(x-0.33*L, 0.75) + kdv_exact(x-0.65*L, 0.4)
## dissipation condition => 0 when working with solitons
mu = 0

T = 150
t = np.linspace(0, T, 501)

solu = diss_kdv_solution(u0, t, L, mu)

plt.figure(figsize=(6,5))
plt.imshow(solu[::-1, :], extent=[0,L,0,T])
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.axis('on')
plt.title('Korteweg DeVries on a Periodic Domain')
plt.show()






