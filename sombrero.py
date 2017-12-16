#!/usr/bin/env python3
# Name: Riley Kendall
# Student ID: 2267883
# Email: kenda106@mail.chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2017
# Assignment: CLASSWORK 12

import numpy as np # Numeric python
import matplotlib.pyplot as plt # Used to plot in python
import numba as nb #enables @nb.jit to be used to speed up code

#one of the coupled first-order ODE's of the duffing oscillator system, satisfying Newton's second law
def dx(y,t):
    return y

#another of the coupled first-order ODE's of the duffing oscillator system, satisfying Newton's second law
def dy(t,y,x,nu,F):
    return ((-nu*y)+x-(x**3)+F*np.cos(t))

@nb.jit
def Newton(a,b,dt,x_initial, y_initial, nu, F):
    '''Newton(a,b,dt,x_initial, y_initial, nu, F)
    Using Newton's Second Law and 4th Order Runge Kutta to solve for the equations describing the duffing oscillator system
    Args:
        a (float) : initial value in time domain
        b (float) : end value in time domain
        dt (float) : time-step size
        x_initial (float) : initial position of the ball in the duffing oscillator system
        y_initial (float) : initial velocity of the ball in the duffing oscillator system
        nu (float) : constant that the velocity in multiplied by to represent damping
        F (float) : driving force on system
    Returns:
        t (array) : The time range considered
        x (array) : The position values of the ball in the duffing oscillator system for time=t
        y (array) : The velocity values of the ball in the duffing oscillator system for time=t'''
    n = int((b-a)/dt) #n values
    t = np.linspace(a,b,n)
    x = np.zeros_like(t)
    x[0]= x_initial
    y = np.zeros_like(t)
    y[0] = y_initial
    for k in range(1,n):
        #4th order RK for y values
        K1y = dy(t[k-1], y[k-1], x[k-1], nu, F)*dt
        K2y = dy(t[k-1] + dt/2, y[k-1] + K1y/2, x[k-1] + K1y/2, nu, F)*dt
        K3y = dy(t[k-1] + dt/2, y[k-1] + K2y/2, x[k-1] + K2y/2, nu, F)*dt
        K4y = dy(t[k-1] + dt, y[k-1] + K3y, x[k-1] + K3y, nu, F)*dt
        y[k] = y[k-1] + (K1y + 2*K2y + 2*K3y + K4y)/6
        #4th order RK for x values
        K1x = dx(y[k-1], t[k-1])*dt
        K2x = dx(y[k-1] + K1x/2, t[k-1] + dt/2)*dt
        K3x = dx(y[k-1] + K2x/2, t[k-1] + dt/2)*dt
        K4x = dx(y[k-1] + K3x, t[k-1] + dt)*dt
        x[k] = x[k-1] + (K1x + 2*K2x + 2*K3x + K4x)/6
    return (t,x,y)

@nb.jit
#Plotting the oscillation motion
def make_plots_oscillation(x,t):
    plt.xlabel('t',fontsize = 16) #labels the x axis
    plt.ylabel('x',fontsize = 16) #labels the y axis
    font = {'size': 16} #adjusts font size
    title = "Oscillation Motion" #titles graph
    plt.title(title)
    plt.plot(x,t,linewidth=1.0,linestyle='-',color='blue')
    plt.show()

@nb.jit
#Plotting the parametric curve for the same time range
def make_plots_parametric(x,y):
    plt.xlabel('x',fontsize = 16) #labels the x axis
    plt.ylabel('y',fontsize = 16) #labels the y axis
    font = {'size': 16} #adjusts font size
    title = "Parametric Curve" #titles graph
    plt.title(title)
    plt.plot(x,y,linewidth=1.0,linestyle='-',color='blue')
    plt.show()

@nb.jit
#Plotting the Poincare section of the parametric curve
def scatter_plot(x,y,n):
    #plt.rc('text', usetex=True)
    ax = plt.subplot
    plt.xlabel('x',fontsize = 16) #labels the x axis
    plt.ylabel('y',fontsize = 16) #labels the y axis
    font = {'size': 16} #adjusts the font size
    title = "Poincare Section" #titles graph
    plt.title(title)
    for k in range(n):
        plt.scatter(x[6283*k],y[6283*k],color="blue") #number 6283 obtained by 314159/50 (where 314159 = length of t and 50=n), since 2pi = t/n according to given equation: t=n2pi
    plt.show()
