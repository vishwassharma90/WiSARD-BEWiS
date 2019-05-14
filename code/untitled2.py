#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:14:06 2019

@author: vishwas
"""

import random

foo = ['a', 'b', 'c', 'd', 'e']
print(random.choice(foo))



my_list = list(range(0,784)) # list of integers from 1 to 99
                              # adjust this boundaries to fit your needs
random.shuffle(my_list)
print(my_list)