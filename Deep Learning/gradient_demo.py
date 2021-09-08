# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from diff_gradient import numerical_gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    
    init_x = np.array([-3.0, 4.0])    

    lr = 0.1
    step_num = 20
    x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
    
    figure, axes = plt.subplots()
    plt.axvline(x=0, ls='--', c='blue')
    plt.axhline(y=0, ls='--', c='blue')
    
    for r in range(6):
        draw_c = plt.Circle(xy=(0,0), radius=r*0.8, ls='dashed', fill=False)
        axes.add_patch(draw_c)
        
        
    plt.plot(x_history[:,0], x_history[:,1], 'o')
    
    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

