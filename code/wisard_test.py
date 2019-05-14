# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from new_jit_WiSARD import WiSARD
import time
import numpy as np
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



#start = time.time()
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
train_with_bleeching = __import__('new_jit_WiSARD').WiSARD.train_with_bleeching
#WiSARD.train_with_bleeching(d,acc_pos,px_train[0:1000],py_train[0:1000])
train_with_bleeching(d,acc_pos,px_train[0:1000],py_train[0:1000])
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
#    end = time.time()
#    print("total time = ",end - start)
