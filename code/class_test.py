#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:12:04 2019

@author: iss
"""

from numba import njit

class MyClass:
    def __init__(self):
        self.k = 1

    def calculation(self):
        k = self.k
        print(2)
        return k

    @staticmethod
    @njit                            
    def complicated(x,k):                                  
        print(0)
        w = 0
        for i in range(500000000):
            w = w
        for a in x:
            b = a**2 + a**3 + k
            print(1)
        return b


if __name__ == "__main__":
    cl = MyClass()
    k = cl.calculation()
    x = [1,2,3]
    cl.complicated(x,k)