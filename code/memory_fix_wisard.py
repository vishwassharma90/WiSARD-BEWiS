#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:19:07 2019

@author: iss
"""

from numba import njit,jit,vectorize,prange
from numba import cuda
import numpy as np
#import struct
import random
import time
import tensorflow as tf
import cv2



def preprocessing(tx_train, ty_train, tx_test, ty_test):
        
    py_train = ty_train.flatten()
    py_test = ty_test.flatten()
    
    px_test = tx_test.reshape(10000,input_size)
    px_train = tx_train.reshape(60000,input_size)
    
    px_train = np.asarray(px_train).astype(np.int32)
    py_train = np.asarray(py_train).astype(np.int32)
    px_test = np.asarray(px_test).astype(np.int32)
    py_test = np.asarray(py_test).astype(np.int32)

    return px_train, py_train, px_test, py_test   


def binaryToDecimal(binary): 
      
    decimal, i = 0, 0
    while(binary != 0): 
        dec = binary % 10
        decimal = decimal + dec * pow(2, i) 
        binary = binary//10
        i += 1
    return decimal
    


class WiSARD:                                                     
    
    def __init__(self,input_size,no_of_rand_pix_selec,nodes,ram_address_count,dis_number):
        self.input_size = input_size
        self.no_of_rand_pix_selec = no_of_rand_pix_selec
        self.nodes = nodes
        self.ram_address_count = ram_address_count
        self.dis_number = dis_number 


    def discriminator(self):
        discriminator = []
#        accumulated_pos = []
        my_list = list(range(0,self.input_size))
        random.shuffle(my_list)
        for i in range(self.dis_number):  #10
            ram = []
            for j in range((int)((self.nodes))): #98    
                table = []
                array = np.zeros((256,1),dtype=np.uint8)
                table.append(array)
                ram.append(table)                                   
            discriminator.append(ram)    
        discriminator = np.asarray(discriminator).astype(np.int32)           #cancel int32 to uint8
        my_list = np.asarray(my_list).astype(np.int32)       #cancel int32 to uint8
        return discriminator, my_list
        
    
    #@staticmethod
    #@njit(parallel = True)
    def train_with_bleeching(self,d,pos,x_train, y_train):
        images = x_train
        lable = y_train    
        
        for i in range(len(images)):
            image = images[i]
            num = lable[i]
            all_ram_of_selected_discriminator = d[num]
            t_ratina = (pos)#[(int)(nodes*num):(int)(nodes*num+nodes)]             #change t_ratina
            
            for i in range((int)(nodes)):
                part = all_ram_of_selected_discriminator[(ram_address_count*i):(ram_address_count*i+ram_address_count)]
                print(part)
                ratina_for_one_ram = t_ratina[i*8:i*8+8]                                #change
                n = []                                                                
                for ix in range(len(ratina_for_one_ram)):
                    pix = ratina_for_one_ram[ix]
                    if image[(pix-1)]>=1:
                        n.append(1)    
                    else:
                        n.append(0)
                
                num = 0
                print(n)
                for i in range(no_of_rand_pix_selec):
                    print(i)
                    num = (n[i])*(10**((no_of_rand_pix_selec-1)-i)) + num
                    print(num)
                
                num  = binaryToDecimal(num)
                print(num)
                address_of_that_ram = (int)(num)
                for key in range(ram_address_count):
                    index = part[key]          
                    if index[0] == address_of_that_ram:
                        index[1] += 1
            



    #@vectorize(['int32(int32,int32,int32,int32)'], target = 'cuda')
    #@staticmethod
    #@njit(parallel = True)
    def test_with_bleaching(self,d,pos,x_test,y_test):
        right = 0
        wrong = 0
        images = x_test
        lable = y_test
        ct = 0.01
        b=1
        
        for i in range(len(images)):
            image = images[i]
            actual_lable = lable[i]
            
            total_sum=[]
            
            for ix in range(dis_number):
                
                t_ratina = (int)(pos)#[int((nodes*ix)):(int)((nodes*ix+nodes))]
                
                sum_of_ram_output = 0
                dis = d[ix]
                
                for i in range(int(nodes)):
                    part = dis[(ram_address_count*i):(ram_address_count*i+ram_address_count)]
                    ratina_for_one_ram = t_ratina[i*8:i*8+7]
                    
                    n = []                                                                
                    for pix in ratina_for_one_ram:
                        if image[(pix-1)]>=1:
                            n.append(1)
                        else:
                            n.append(0)
                    
                    num = 0
                    for i in range(no_of_rand_pix_selec):
                        num = (n[i])*(10**((no_of_rand_pix_selec-1)-i)) + num
                    
                    address_of_that_ram = (int)(num)
                
                    for key in range(len(part)):
                        prt = part[key]
                        if prt[0] == address_of_that_ram and prt[1]>=b:           
                            sum_of_ram_output += 1
                
                total_sum.append(sum_of_ram_output)        
        
            max_sum = 0
            sec_max = 0
            idx = 0
            
            for i in range(len(total_sum)):
                if max_sum < total_sum[i]:
                    max_sum = total_sum[i]
                    idx = i
                    
            for j in range(len(total_sum)):
                if sec_max < total_sum[j] and j!=idx:
                    sec_max = total_sum[j]
                    
            index_of_dis = idx
            if index_of_dis == actual_lable:
                right += 1
            else:
                wrong += 1
            
            if max_sum == sec_max or max_sum == 0:
                confidence = 0
            else:
                confidence = 1 - float(sec_max)/float(max_sum)
            if confidence < ct:
                b += 1
        
        return right,wrong
    
    
    
    
    
    
input_size = 28*28
no_of_rand_pix_selec = 2**(3)     ## ** (must) no_of_rand_pix_selec = 2^(n) where n is 0,1,2...
nodes = int(input_size/no_of_rand_pix_selec)    #98
ram_address_count = 2**(no_of_rand_pix_selec)   #256
dis_number = 10                #10 i.e number of lables

(tx_train, ty_train), (tx_test, ty_test) = tf.keras.datasets.mnist.load_data()
px_train, py_train, px_test, py_test = preprocessing(tx_train, ty_train, tx_test, ty_test)


w = WiSARD(input_size,no_of_rand_pix_selec,nodes,ram_address_count,dis_number)
d, acc_pos = w.discriminator()
#print(d)
print(acc_pos)

starttrain = time.time()
w.train_with_bleeching(d,acc_pos,px_train[0:1000],py_train[0:1000])
endtrain = time.time()
print("time train = ",endtrain - starttrain)

starttest = time.time()
right,wrong = w.test_with_bleaching(d,acc_pos,px_test[0:100],py_test[0:100])
endtest = time.time()
print("time test = ",endtest - starttest)

print("number of right result = ",right)
print("number of wrong results = ",wrong)
accuracy = ((right)/(right+wrong))*100
print("accuracy by testing the model =",accuracy)