# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import struct
import time
from numba import njit,jit,vectorize,autojit
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


#@njit
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result
@njit
def conc_list(list1):
    num = 0
    for i in range(8):
        num = (list1[i])*(10**(7-i)) + num
    return num    



def discriminator():
#    tt=0
    discriminator = []
    accumulated_pos = []
    my_list = list(range(0,784))
    for i in range(10):  #10
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
            positions = my_list[j*8:j*8+8]                            #with this we can access every pixels of the image
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
        
        
        di = []
        for j in range(len(ram)):
            for i in range(len(ram[j])):
                for key, value in ram[j][i].items():
                    temp = [key,value]
#                    print(temp)
                    di.append(temp)
        
#        print(1)
        
        discriminator.append(di)
#    if tt <1:
#        print(discriminator)
#        tt = tt + 1
    #print(accumulated_pos[0])
    return discriminator, accumulated_pos
        

   
#@vectorize(['int32(int32,int32,int32,int32)'], target = 'cuda')
#@cuda.jit
@njit
def train2(i,all_ram_of_selected_discriminator,t_ratina,image):
    part = all_ram_of_selected_discriminator[(256*i):(256*i+256)]
    ratina_for_one_ram = t_ratina[i]

    n = []                                                                
    for ix in range(len(ratina_for_one_ram)):
        pix = ratina_for_one_ram[ix]
        if image[(pix-1)]>=1:
            n.append(1)
        else:
            n.append(0)
        
                    
    address_of_that_ram = (int)(conc_list(n))
    for key in range(256):
        index = part[key]
        train3(index,address_of_that_ram)              
            

@njit
def train3(index,address_of_that_ram):
    if index[0] == address_of_that_ram:
        index[1] += 1
        
        
@njit
def train_discriminator_with_bleaching(d,pos,x_train, y_train):
    
    images = x_train
    lable = y_train    
    
    for i in range(len(images)):
#        print(image)
        image = images[i]
        num = lable[i]
        all_ram_of_selected_discriminator = d[num]
        #print(all_ram)
        t_ratina = pos[(98*num):(98*num+98)]
        #print(t_ratina)
        
        
        for i in range(98):
            train2(i,all_ram_of_selected_discriminator,t_ratina,image)
            
        


#@vectorize(['int32(int32,int32,int32,int32)'], target = 'cuda')
@njit
def test1(d,pos,x_test,y_test):
    right = 0
    wrong = 0
    images = x_test
    lable = y_test
    
    
    for i in range(len(images)):
        image = images[i]
        actual_lable = lable[i]
     
        total_sum = test2(pos,d,image)
     


        max_sum = 0
        idx = 0
        for i in range(len(total_sum)):
            if max_sum < total_sum[i]:
                max_sum = total_sum[i]
                idx = i
        index_of_dis = idx
        if index_of_dis == actual_lable:
            right += 1
            #print(1)
        else:
            wrong += 1
            #print(0)     
            
    print(1)
    return right,wrong



@njit
def test2(pos,d,image):
    total_sum=[]
    for ix in range(10):
        t_ratina = pos[(98*ix):(98*ix+98)]
        sum_of_ram_output = 0
        dis = d[ix]
        
        for i in range(98):
            part = dis[(256*i):(256*i+256)]
            ratina_for_one_ram = t_ratina[i]
            
            n = []                                                                
            for pix in ratina_for_one_ram:
                if image[(pix-1)]>=1:
                    n.append(1)
                else:
                    n.append(0)
                #print(n)
                
            address_of_that_ram = (int)(conc_list(n))
                
            for key in range(len(part)):
                prt = part[key]
                if prt[0] == address_of_that_ram and prt[1]>=1:
                    sum_of_ram_output += 1
            
        total_sum.append(sum_of_ram_output)        
    return total_sum
    

if __name__ == "__main__":

    # the mnist has 28*28(784) pixels that we need to train in the discriminator
    # we have to initialize 10 discriminator for 0 to 9 number images(one discriminator for one pattern)
    # We are taking 
    #
    
    input_size = 28*28
    no_of_rand_pix_selec = 8   
    nodes = input_size/no_of_rand_pix_selec    #98
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
    
    #d = []
    d, acc_pos = discriminator()
    
    d = np.asarray(d).astype(np.int32)
#    print(d)
#    
    print(type(d))
    
    acc_pos = np.asarray(acc_pos).astype(np.int32)
    print(type(acc_pos))
    
    #print(d[0][0])
    #print(acc_pos)
    #print(x_train)
    (tx_train, ty_train), (tx_test, ty_test) = tf.keras.datasets.mnist.load_data()
    px_train, py_train, px_test, py_test = preprocessing(tx_train, ty_train, tx_test, ty_test)
    
    #print(px_train)
    
#    print(px_test[0].shape)
    px_train = np.asarray(px_train).astype(np.int32)
    py_train = np.asarray(py_train).astype(np.int32)
    px_test = np.asarray(px_test).astype(np.int32)
    py_test = np.asarray(py_test).astype(np.int32)

#    
#    d_d = cuda.to_device(d)
#    acc_pos_d = cuda.to_device(acc_pos)
#    px_train_d = cuda.to_device(px_train)
#    py_train_d = cuda.to_device(py_train)
#    threads_per_block = 128
#    blocks_per_grid = 32
    
    
    
    
    starttrain = time.time()
    train_discriminator_with_bleaching(d,acc_pos,px_train[0:60000],py_train[0:60000])
#    cuda.synchronize()
    endtrain = time.time()
    print("time train = ",endtrain - starttrain)
    #print(view)
    #print(x_test[1])
    print('this is d[0] =',d[0])
    
    
#    d_d = cuda.to_device(d)
#    acc_pos_d = cuda.to_device(acc_pos)
#    px_test_d = cuda.to_device(px_test)
#    py_test_d = cuda.to_device(py_test)
#    threads_per_block = 128
#    blocks_per_grid = 32
    
    
    starttest = time.time()
    right,wrong = test1(d,acc_pos,px_test[0:10000],py_test[0:10000])
#    right,wrong = test2(d,acc_pos,px_test[0:100],py_test[0:100])
#    right,wrong = test3(d,acc_pos,px_test[0:100],py_test[0:100])
#    right,wrong = test4(d,acc_pos,px_test[0:100],py_test[0:100])
    #cuda.synchronize()
    endtest = time.time()
    print("time test = ",endtest - starttest)
    print("number of right result = ",right)
    print("number of wrong results = ",wrong)
    
    accuracy = ((right)/(right+wrong))*100
    print("accuracy by testing the model =",accuracy)
    
    
#    print(x_test.shape)
#    print(y_test.shape)
    
#    print(ty_test.shape)
#    ty_test.flatten()
#    print(ty_test)
#    print(tx_train[0].flatten().shape)
#    print(tx_train[0].flatten())