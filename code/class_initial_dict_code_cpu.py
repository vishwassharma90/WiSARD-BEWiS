#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:57:03 2019

@author: iss
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import struct
import time
from numba import njit,jit,vectorize
from numba import cuda
import numpy as np
#import struct
import random
#import matplotlib
import tensorflow as tf                         #i have tensorflow in the system, the tensorflow environment is also there
                                                     #i am able to activate it in terminal by (conda activate tensorflow_gpuenv)
                                                     #i am able to import it in gedit python file after that          
                                                     #but i am not able to import it in the anacoda(in this file)


#def readFiles(labelsFilePath, imagesFilePath):
#    flImg = open(imagesFilePath, 'rb')
#    flLbl = open(labelsFilePath, 'rb')
#    (mNumberImg, sizeImg, height, width) = struct.unpack('>IIII', flImg.read(16))
#    (mNumberLbl, sizeLbl) = struct.unpack('>II', flLbl.read(8))
#    imgs = []
#    labels = map(ord, flLbl.readlines()[0])
#    for i in range(sizeImg):
#        imgs.append({'img': map(ord, list(flImg.read(width*height))), 'label': labels[i]})
#    flImg.close()
#    flLbl.close()
#    return imgs

#@vectorize
#@jit
#@jit                                                     
def preprocessing(tx_train, ty_train, tx_test, ty_test):
    
    py_train = ty_train.flatten()
    py_test = ty_test.flatten()
        
#    px_test = [l.tolist() for l in tx_test]
#    px_train = [l.tolist() for l in tx_train]
#    print(px_test) 
    
    px_test = tx_test.reshape(10000,784)
    px_train = tx_train.reshape(60000,784)
    #print(py_test)
    #print(px_train.shape)
    
    return px_train, py_train, px_test, py_test

class WiSARD:
    
    def __init__(self,input_size,no_of_rand_pix_selec,nodes,dis_number):
        self.input_size = input_size
        self.no_of_rand_pix_selec = no_of_rand_pix_selec
        self.nodes = nodes
        self.dis_number = dis_number
    
    
    #@jit
    def concatenate_list_data(self,list):
        result= ''
        for element in list:
            result += str(element)
        return result

    #@jit
    def discriminator(self):
        #    tt=0
        discriminator = []
        accumulated_pos = []
        my_list = list(range(0,input_size))
        for i in range(dis_number):  #10
            #print(1)
            ram = []
            #n_0 = [1990]
            #        taken = []
            random.shuffle(my_list)                                      #with this we can access every pixels of the image
            #        n_0 =[]
            for j in range((int)((nodes))): #98    
                
                total_pos = []
                #            total_pos1 = []
                positions = []
                
                #for k in range(no_of_rand_pix_selec):  #8
                positions = my_list[j*no_of_rand_pix_selec:j*no_of_rand_pix_selec+no_of_rand_pix_selec]                            #with this we can access every pixels of the image
                #print(positions)
                #            for k in range(no_of_rand_pix_selec):  #8
                ##                                               # this is wrong as for every ram in a discriminator this will reset
                #                n = random.randint(1, input_size)                     #and we want it to reset after 98 ram of one discriminator
                #                                                                          #but computer hange if i do it. so check later
                #                for i in n_0:
                #                    if n != i:
                #                        positions.append(n)    
                
                #            n_0.append(n)
                #            print(n_0)
                #                if taken.count(n) == 0:
                #                    positions.append(n)                                                          
                    #taken.append(n)                                                          
                    
                    #if tt < 1:
                    #    print(positions)
                    #    tt = tt + 1
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
                    #table.append({x: 0})    
                table.append(dictionary)
            
            #            if tt < 1:
            #                print(table)
                
                ram.append(table)
            
            discriminator.append(ram)
            #    if tt <1:
            #        print(discriminator)
            #        tt = tt + 1
            #print(accumulated_pos[0])
        return discriminator, accumulated_pos
        
    #@jit        
    def train(self,d,pos,x_train, y_train):
        
        images = x_train
        lable = y_train    
        
        for i,image in enumerate(images):
            #        print(image)
            l = lable[i]
            num = l  #have to check how to do it after i do the preprocessing part and see the dataset lable type
            all_ram_of_selected_discriminator = d[num]
            #print(all_ram)
            t_ratina = pos[(nodes*num):(nodes*num+nodes)]
            #print(t_ratina)
            for x,r in enumerate(all_ram_of_selected_discriminator):     #here i want to iterate the ram of the perticular discriminator
                #print(x,r)
                #pattern_to_con = total_pos[r]    # here i want to take only the positions of the node of that discriminator
                #print(x,r)
                ratina_for_one_ram = t_ratina[x]
                #print(ratina_for_one_ram)
                #for i in (0,len(ratina_for_one_ram)):
                #   print(i)
                threshold = 0         # this define the threshold. right now if any one pixel in ratina(8) is >=1 then the value
                n = []                                                              #saved is 1. if we want to chnage it we can     
                for pix in ratina_for_one_ram:                                          #like if value of 5 pixel is>=1 then save 1
                    if image[(pix-1)]>=1:
                        n.append(1)
                        threshold = threshold + 1
                    else:
                        n.append(0)
                        #print(n)
                        
                address_of_that_ram = self.concatenate_list_data(n)
                #print(address_of_that_ram)
                #print(threshold)
                
                if threshold >= 1:                  #refer above comment
                    for index,key in enumerate(r[0]):
                        #print(1)
                        #print(key)
                        if key == address_of_that_ram:
                            r[0][key] += 1
                            #print(key)
                            #print(index)
                            #print(2)
#                       r[address_of_that_ram] += 1
#                       print(address_of_that_ram)
#                       print(r[address_of_that_ram])
#               else:
#                   print()
            #   print(x,r[0])
            #   address_of_one_ram = ','.join(n)
            #   print(address_of_one_ram)
    
        #return 1
        #@jit
    def test(self,d,pos,x_test,y_test):
        right = 0
        wrong = 0
        images = x_test
        lable = y_test
        non_rec = 0
        
        for i,image in enumerate(images):
            actual_lable = lable[i]
            #sum_of_ram_output = []
            #        print(image)
            total_sum=[]
            #        total_sum_of_all_ram=[]
            for index,dis in enumerate(d):
                t_ratina = pos[(nodes*index):(nodes*index+nodes)]
                
                sum_of_ram_output = 0
                for x,r in enumerate(dis):
                    ratina_for_one_ram = t_ratina[x]
                    
                    n = []                                                                
                    for pix in ratina_for_one_ram:
                        if image[(pix-1)]>=1:
                            n.append(1)
                        else:
                            n.append(0)
                            #print(n)
                            
                    address_of_that_ram = self.concatenate_list_data(n)
                    #print(address_of_that_ram)
                    
                    
                    for index,key in enumerate(r[0]):
                        if key == address_of_that_ram and r[0][key]>=1:
                            #print(111)
                            #print(key,r[0][key])
                            sum_of_ram_output += 1
                            
                            
                total_sum.append(sum_of_ram_output)
                #        print(1)
                #        print(total_sum)
                
            if max(total_sum) >= 1:
                index_of_dis = total_sum.index(max(total_sum))
                if index_of_dis == actual_lable:
                    right += 1
                    #print(1)
                else:
                    wrong += 1
                    #print(0)
                    
            else:
                #wrong += 1
                non_rec += 1
    
#    print(1)    
#    print("non recognized images = ", non_rec)
        
        
        return right,wrong



if __name__ == "__main__":

    # the mnist has 28*28(784) pixels that we need to train in the discriminator
    # we have to initialize 10 discriminator for 0 to 9 number images(one discriminator for one pattern)
    # We are taking 
    #
#    start = time.time()
    input_size = 28*28
    no_of_rand_pix_selec = 2**(3)     ## ** (must) no_of_rand_pix_selec = 2^(n) where n is 0,1,2... 
    nodes = int(input_size/no_of_rand_pix_selec)    #98
    ram_address_count = 2**(no_of_rand_pix_selec)#256
    dis_number = 10                #10 i.e number of lables
    #print(nodes)
    #n = random.randint(1, input_seize)
    #print(n)
    
#    imagesTrainingFile = 'train-images-idx3-ubyte'
#    labelsTrainingFile = 'train-labels-idx1-ubyte'
#    imagesTestFile = 't10k-images-idx3-ubyte'
#    labelsTestFile = 't10k-labels-idx1-ubyte'
#    
#    training = readFiles(labelsTrainingFile, imagesTrainingFile)
#    
#    for i in range(3):
#        print(training[i])
    
    x_train1 = np.random.randint(10, size=784)
    x_train2 = np.random.randint(10, size=784)
    x_train3 = np.random.randint(10, size=784)
    x_train = [x_train1,x_train2,x_train3]
    y_train = [0,1,2]    
    #print(x_train[0])
    
    
    x_test1 = np.random.randint(10, size=784)
    #print(x_test1)
    x_test2 = np.zeros(784)                    #use this for accuracy = 0 check
    x_test = [x_train1,x_test1,x_test2]                             #use this for accuracy = 100 check
#    print(x_test)
    y_test = [0,1,2]
    #print(x_test[0],x_test1)
    
    w = WiSARD(input_size,no_of_rand_pix_selec,nodes,dis_number)
    d, acc_pos = w.discriminator()
    
    print(type(d))
    
    #print(d[0][0])
    #print(acc_pos)
    #print(x_train)
    (tx_train, ty_train), (tx_test, ty_test) = tf.keras.datasets.mnist.load_data()
    px_train, py_train, px_test, py_test = preprocessing(tx_train, ty_train, tx_test, ty_test)
    
    #print(px_train)
    
#    print(px_test[0].shape)
    
    starttrain = time.time()
    train_the_network = w.train(d,acc_pos,px_train[0:5000],py_train[0:5000])
    endtrain = time.time()
    print("time train = ",endtrain - starttrain)
    #print(view)
    #print(x_test[1])
    #print(d[0][0])
    starttest = time.time()
    right,wrong = w.test(d,acc_pos,px_test[0:1000],py_test[0:1000])
    endtest = time.time()
    print("time test = ",endtest - starttest)
    print("number of right result = ",right)
    print("number of wrong results = ",wrong)
    
    accuracy = ((right)/(right+wrong))*100
    print("accuracy by testing the model =",accuracy)
#    end = time.time()
#    print("total time = ",end - start)
    
#    print(x_test.shape)
#    print(y_test.shape)
    
#    print(ty_test.shape)
#    ty_test.flatten()
#    print(ty_test)
#    print(tx_train[0].flatten().shape)
#    print(tx_train[0].flatten())
#    print(timeit.timeit(discriminator())) 