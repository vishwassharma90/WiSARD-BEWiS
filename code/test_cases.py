#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:14:54 2019

@author: iss
"""

from numba import njit,jit,vectorize
from numba import cuda
import numpy as np
#import struct
import random
import time

#@vectorize(['int32(int32)'], target = 'cuda')
#@cuda.jit
#@jit
#def addition(d):
#    
#    
#    for i in range(50000000):
#        i +=1
#    
#    return d

@njit
def f(w):
    return w

#@vectorize(['int32(int32)'], target = 'cuda')
@njit 
def actual_addition(d,w):
#    for i in range(500000000):
    dam = d
    dam += dam
    print(dam)
    for i in range(500000000):
        w.append(2)
#    for i,x in enumerate(w):
#        print(x)
    f(w)    
    return d

@cuda.jit
def add1(d,w):
    
    x,y = cuda.grid(2) 
    d[x][y] = d[x][y] * 20
    #print(d[x][y])
    #actual_addition(d)
    #return d
#    for i in range(2):
#        print(w)
#    
if __name__ == "__main__":
    
    d0 = [1,2,3,6357,467,467,54]
    
#    d = [[[{'23':1 , '2':32},{'24':8 , '35':2}],[[647,5],[456,5],[35,4]]]]
    
    my_dict = [[{'00001' : 1, '24324' : 0},{'13' : 1, '24' : 0}],[{'13124' : 1, '24324' : 0},{'13' : 1, '24' : 0}]]
    di = []
    for j in range(len(my_dict)):
        for i in range(len(my_dict[j])):
            for key, value in my_dict[j][i].items():
                temp = [key,value]
                di.append(temp)
    
    di = np.asarray(di).astype(np.int32)
    #print(di)
    w = [1,2,3]
    w_d = cuda.to_device(w)
    ad = actual_addition(di,w)
#    di = di.flatten()
##    print(di)
#    di_d = cuda.to_device(di)
#    threads_per_block = 16
#    blocks_per_grid = 1
#    
#    starttest = time.time()
#    add1[blocks_per_grid,threads_per_block](di_d,w_d)
#    cuda.synchronize()
#    endtest = time.time()
#    print("time test = ",endtest - starttest)
#    print(di_d)
#    print(di_d.copy_to_host())
    
#    print(type(d0))
#    print(type(d))
#    d = np.asarray(d).astype(np.int32)
#    #print(type(d[0][0][0][0]))
#    addi = addition(d)
#    print(addi)
a = np.array([[1, 2], [3, 4]])
for index, x in np.ndenumerate(a):
    print(index, x)
print(2 * 10**2)
num = 0
list1 = [1,2,3,4,5,6,7,8]
for i in range(8):
    print(i)
    num = (list1[i])*(10**(7-i)) + num
    print(num)
print(num)    


