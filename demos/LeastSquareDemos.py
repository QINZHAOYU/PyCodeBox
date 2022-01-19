# -*- encoding: utf-8 -*-
# -------------------------------------------------------------
'''
@Copyright :  Copyright(C) 2021, Qin ZhaoYu. All rights reserved. 

@Author    :  Qin ZhaoYu   
@See       :  https://github.com/QINZHAOYU

@Desc      :  最小二乘法案例。  

Change History:
---------------------------------------------------------------
v1.0, 2022/01/19, Qin ZhaoYu, zhaoyu.qin@foxmail.com
Init model.
'''
# -------------------------------------------------------------
import numpy as np
from numpy.linalg import lstsq  # least square method
import pandas as pd 
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt


def demo1():
    ''' y = k*x + b '''
    x = np.array([0, 1, 2, 3])
    y = np.array([-1, 0.2, 0.9, 2.1])

    '''
    | x1 1 |            | y1 | 
    | x2 1 |    | k |   | y2 |  
    | x3 1 |  * | b | = | y3 | 
    | x4 1 |            | y4 | 
       A          X       B
    '''
    A = np.vstack([x, np.ones(len(x))]).T
    B = y

    coeff, err, rank, singulars = lstsq(A, B, rcond=None)
    print(coeff)
    print("residual sum of squares: {}, rank: {}".format(err, rank))

    k, b = coeff
    y2 = [k*i + b for i in x]
    plt.plot(x, y, ls="-", lw=2, c="red", label="measure")
    plt.plot(x, y2, ls="-", lw=2, c="green", label="simu")
    plt.legend()
    plt.show()


def demo2():
    ''' 
    y1 = k1 * x + b1 
    y2 = k2 * x + b2
    '''
    x = np.array([0, 1, 2, 3])
    y1 = np.array([-1, 0.2, 0.9, 2.1])
    y2 = [y+2 for y in y1]

    '''
    | x1 1 |                | y1_1  y2_1| 
    | x2 1 |    | k1 k2 |   | y1_2  y2_2|  
    | x3 1 |  * | b1 b2 | = | y1_3  y2_3| 
    | x4 1 |                | y1_4  y2_4| 
       A            X             B
    '''
    A = np.vstack([x, np.ones(len(x))]).T
    B = np.vstack([y1, y2]).T  # two columns.

    coeff, err, rank, singulars = lstsq(A, B, rcond=None)
    print(coeff)
    print("residual sum of squares: {}, rank: {}".format(err, rank))

    k, b = coeff[0][0], coeff[1][0]  # first column as coeff.
    y1_2 = [k*i + b for i in x] 
    plt.plot(x, y1, ls=':', lw=2, c='b', marker='o', markerfacecolor='r', label="measure")
    plt.plot(x, y1_2, ls='-', lw=2, c='r',  label="simu1")
    plt.legend()
    plt.show()

    k, b = coeff[0][1], coeff[1][1] # second column as coeff.
    y2_2 = [k*i + b for i in x] 
    plt.plot(x, y2, ls=':', lw=2, c='b', marker='o', markerfacecolor='r', label="measure")
    plt.plot(x, y2_2, ls='-', lw=2, c='r',  label="simu1")
    plt.legend()
    plt.show()


def demo3():
    ''' y = k1  *x^2 + k2 * x + b '''
    k1_true, k2_true, b_true = 3, 2, 1  

    x = np.linspace(-1, 1, 100)
    y_true = k1_true * x**2 + k2_true * x + b_true

    xi = 1 - 2 * np.random.rand(100)
    yi = k1_true * xi**2 + k2_true * xi + b_true + np.random.randn(100)

    '''
    | x1^2    x1    1 |    | k1 |   | y1   | 
    | x2^2    x2    1 |    | k2 |   | y2   |  
    | ............... |  * | b  | = | .... | 
    | x100^2  x100  1 |             | y100 | 
            A                X        B
    '''
    A = np.vstack([xi**2, xi**1, xi**0]).T
    B = yi

    coeff, err, rank, singulars = lstsq(A, B, rcond=None)
    print(coeff)
    print("residual sum of squares: {}, rank: {}".format(err, rank))

    k1, k2, b = coeff
    y2 = [k1 * i**2 + k2 * i + b for i in x]
    plt.plot(xi, yi, 'go', alpha=0.5, label="nosie")
    plt.plot(x, y_true, ls="-", lw=2, c="red", label="math")
    plt.plot(x, y2, ls="-", lw=2, c="green", label="simu")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # demo1()
    # demo2()
    demo3()
