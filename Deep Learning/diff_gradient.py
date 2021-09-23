# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 21:11:42 2021

@author: user
"""
import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 函數微分 中央分差
def num_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)

# 函數1
def function_1(x):
    return x ** 2 + x

# 函數2 --> 偏微分
def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)
    
def function_z(x0, x1):
    return x0**2 + x1**2
    
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val  # 値を元に戻す
        
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad


# 微分切線方程式: y = f(a) + d(x − a) = f(a) - d * a + d * x
# 這部分回傳一個function讓我們可以得到函數再把x帶入.
def tangent_fun(f, a):
    d = num_diff(f, a)
    #print(d)
    y = f(a) - d * a
    return lambda x: d * x + y

def draw_tangent_line():
    x = np.arange(0.0, 20, 00.1)
    y = function_1(x)

    tan_fun = tangent_fun(function_1, 5)
    y2 = tan_fun(x)

    plt.plot(x, y2, label = "tangent line")
    plt.plot(x, y, label = "line")
    plt.xlabel("$x$")
    plt.ylabel("$f (x)$")
    plt.legend()
    plt.show()

def draw_gradient():
    x0 = np.arange(-2.0, 2.5, 0.25)
    x1 = np.arange(-2.0, 2.5, 0.25)
    
    #將x和y擴展當相同大小.再轉一維
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    #求梯度
    grad = numerical_gradient(function_2, np.array([X, Y]).T).T
    
    print('-------- X ----------------')
    print(X)
    print('-------- Y ----------------')
    print(Y)
    print('-------- Gradient ----------------')
    print(grad)
    
    #畫出quiver方向圖, xlim指定顯示大小, label標籤顯示, grid顯示網格
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()
    
def draw_mutivar_3d():
    x0 = np.arange(-2.0, 2.5, 0.25)
    x1 = np.arange(-2.0, 2.5, 0.25)
    
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    Z = function_2(np.array([X, Y]).T).T
    
    # plt.figure(figsize = (9.6, 7.2))
    ax = plt.axes(projection='3d')
    surf = ax.plot_trisurf(X, Y, Z, cmap='RdGy')
    
    plt.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$x0$')
    ax.set_ylabel('$x1$')
    ax.set_zlabel('$y = x_0^2 * x_1^2$', )
    # ax.set_xlim(-2.5, 2.7)
    # ax.set_ylim(-2.5, 2.7)
    
    plt.show()
    
def draw_mutivar_3ds():
    x0 = np.arange(-2.0, 2.5, 0.1)
    x1 = np.arange(-2.0, 2.5, 0.1)
    
    X, Y = np.meshgrid(x0, x1)
    Z = function_z(X, Y)
    
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='RdGy')
    
    plt.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('$x0$')
    ax.set_ylabel('$x1$')
    ax.set_zlabel('$y = x_0^2 * x_1^2$', )
    
    plt.show()


if __name__ == '__main__':
    
    plt.rcParams['figure.figsize'] = (9.6, 7.2)
    # draw_tangent_line()
    # draw_gradient()
    draw_mutivar_3d()
    draw_mutivar_3ds()
    
    
   