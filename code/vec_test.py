#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:03:21 2019

@author: iss
"""
import numpy as np
from numba import njit,vectorize
from numpy import exp
import time
from numba import cuda


n = 1000000

greyscales = np.floor(np.random.uniform(0, 255, n).astype(np.float32))
weights = np.random.normal(.5, .1, n).astype(np.float32)





normalized = np.empty_like(greyscales)
weighted = np.empty_like(greyscales)
activated = np.empty_like(greyscales)

@vectorize(['float32(float32)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32,float32)'], target='cuda')
def weigh(values, weights):
    return values * weights

@vectorize(['float32(float32)'], target='cuda')
def activate(values):
    return ( np.exp(values) - np.exp(-values) ) / ( np.exp(values) + np.exp(-values) )




start = time.time()
greyscales_d = cuda.to_device(greyscales)
weights_d = cuda.to_device(weights)
normalized = normalize(greyscales_d)
weighted = weigh(normalized, weights_d)
SOLUTION = activate(weighted)

end = time.time()
print("time = ",end - start)