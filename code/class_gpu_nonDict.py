# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import time
from numba import njit
import numpy as np
import random
import tensorflow as tf


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


class WiSARD:                                                     
    
    def __init__(self,input_size,no_of_rand_pix_selec,nodes,ram_address_count,dis_number):
        self.input_size = input_size
        self.no_of_rand_pix_selec = no_of_rand_pix_selec
        self.nodes = nodes
        self.ram_address_count = ram_address_count
        self.dis_number = dis_number 


    def discriminator(self):
        discriminator = []
        accumulated_pos = []
        my_list = list(range(0,input_size))
        for i in range(dis_number):  #10
            ram = []
            random.shuffle(my_list)
            for j in range((int)((nodes))): #98    
                total_pos = []            
                positions = []
                positions = my_list[j*no_of_rand_pix_selec:j*no_of_rand_pix_selec+no_of_rand_pix_selec]
                accumulated_pos.append(positions)
                total_pos = np.vstack(positions)
                #print(total_pos)
                table = []
                dictionary = {}
                
                max = len("{0:b}".format(2**len(total_pos))) - 1
                for i in range(2**len(total_pos)):
                    x = (('0' * max) + "{0:b}".format(i))
                    x = x[len(x)-max:]
                    dictionary[x] = 0
                table.append(dictionary)
                #print(table)
                ram.append(table)
        
            di = []
            for j in range(len(ram)):
                for i in range(len(ram[j])):
                    for key, value in ram[j][i].items():
                        temp = [key,value]
                        di.append(temp)
                        
            discriminator.append(di)
    
        discriminator = np.asarray(discriminator).astype(np.int32)
        accumulated_pos = np.asarray(accumulated_pos).astype(np.int32)
        return discriminator, accumulated_pos
        
    
    @staticmethod
    @njit
    def train(d,pos,x_train, y_train):
        
        images = x_train
        lable = y_train    
        
        for i in range(len(images)):
            image = images[i]
            num = lable[i]
            all_ram_of_selected_discriminator = d[num]
            t_ratina = pos[(int)(nodes*num):(int)(nodes*num+nodes)]
            
            for i in range((int)(nodes)):
                part = all_ram_of_selected_discriminator[(ram_address_count*i):(ram_address_count*i+ram_address_count)]
                ratina_for_one_ram = t_ratina[i]
                
                #threshold = 0               
                n = []                                                                
                for ix in range(len(ratina_for_one_ram)):
                    pix = ratina_for_one_ram[ix]
                    if image[(pix-1)]>=1:
                        n.append(1)
                        #threshold = threshold + 1     
                    else:
                        n.append(0)
                
                num = 0
                for i in range(no_of_rand_pix_selec):
                    num = (n[i])*(10**((no_of_rand_pix_selec-1)-i)) + num
            
                address_of_that_ram = (int)(num)
                for key in range(ram_address_count):
                    index = part[key]
                    #if threshold >= 1:          
                    if index[0] == address_of_that_ram:
                        index[1] += 1
            

    #@vectorize(['int32(int32,int32,int32,int32)'], target = 'cuda')
    @staticmethod
    @njit
    def test(d,pos,x_test,y_test):
        right = 0
        wrong = 0
        images = x_test
        lable = y_test
        
        
        for i in range(len(images)):
            image = images[i]
            actual_lable = lable[i]
            
            total_sum=[]
            
            for ix in range(dis_number):
                
                t_ratina = pos[int((nodes*ix)):(int)((nodes*ix+nodes))]
                
                sum_of_ram_output = 0
                dis = d[ix]
                
                for i in range(int(nodes)):
                    part = dis[(ram_address_count*i):(ram_address_count*i+ram_address_count)]
                    ratina_for_one_ram = t_ratina[i]
                    
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
                        if prt[0] == address_of_that_ram and prt[1]>=1:
                            sum_of_ram_output += 1
                
                total_sum.append(sum_of_ram_output)        
        
            max_sum = 0
            idx = 0
            for i in range(len(total_sum)):
                if max_sum < total_sum[i]:
                    max_sum = total_sum[i]
                    idx = i
            index_of_dis = idx
            if index_of_dis == actual_lable:
                right += 1
            else:
                wrong += 1
                
        return right,wrong


if __name__ == "__main__":
#    start = time.time()
    input_size = 28*28
    no_of_rand_pix_selec = 2**(3)     ## ** (must) no_of_rand_pix_selec = 2^(n) where n is 0,1,2...
    nodes = int(input_size/no_of_rand_pix_selec)    #98
    ram_address_count = 2**(no_of_rand_pix_selec)   #256
    dis_number = 10                #10 i.e number of lables
    
    
    (tx_train, ty_train), (tx_test, ty_test) = tf.keras.datasets.mnist.load_data()
    px_train, py_train, px_test, py_test = preprocessing(tx_train, ty_train, tx_test, ty_test)
    
    
    w = WiSARD(input_size,no_of_rand_pix_selec,nodes,ram_address_count,dis_number)
    d, acc_pos = w.discriminator()
    print(d.shape)
    
    starttrain = time.time()
    w.train(d,acc_pos,px_train[0:60000],py_train[0:60000])
    endtrain = time.time()
    print("time train = ",endtrain - starttrain)
    
    
    starttest = time.time()
    right,wrong = w.test(d,acc_pos,px_test[0:10000],py_test[0:10000])
    endtest = time.time()
    print("time test = ",endtest - starttest)
    
    
    print("number of right result = ",right)
    print("number of wrong results = ",wrong)
    accuracy = ((right)/(right+wrong))*100
    print("accuracy by testing the model =",accuracy)
#    end = time.time()
#    print("total time = ",end - start)