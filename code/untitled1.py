#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:52:16 2019

@author: vishwas
"""

import numpy as np 


x_trian11 = np.random.randint(10, size=784)
#print(x_trian11)





def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

print(concatenate_list_data([0, 5, 12, 2]))
print(int(concatenate_list_data([0, 5, 12, 2])))