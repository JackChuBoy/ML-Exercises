# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 23:47:51 2021

@author: user
"""
import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)

def elu(x, alpha=1):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def selu(x):
    scale = 1.0507
    alpha = 1.67326
    return scale * elu(x, alpha)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def step_function(x):
    return np.array(x >= 0, dtype=np.int)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def plot_tanh(x):
    y = tanh(x)
    plt.plot(x, y)
    plt.ylim(-1.1, 1.1)
    plt.show()

def plot_relu(x):
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.5)
    plt.show()

def plot_elu(x):
    y = elu(x)
    plt.plot(x, y)
    plt.ylim(-2.0, 5.5)
    plt.show()
    
def plot_selu(x):
    y = selu(x)
    plt.plot(x, y)
    plt.ylim(-2.0, 5.5)
    plt.show()

def plot_step_function(x):
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show() 
    
def plot_compare_sig_step(x):
    y1 = sigmoid(x)
    y2 = step_function(x)

    plt.plot(x, y1)
    plt.plot(x, y2, 'k--')
    plt.ylim(-0.1, 1.1)
    plt.show()

def plot_sigmoid(x):
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()

if __name__ == '__main__':

    x = np.arange(-5.0, 5.0, 0.1)
    plot_step_function(x)
    plot_sigmoid(x)
    plot_relu(x)
    plot_compare_sig_step(x)
    plot_elu(x)
    plot_selu(x)
    plot_tanh(x)