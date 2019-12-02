import numpy as np
import cv2
import itertools
import time
from numba import njit,jit,prange,vectorize,int32,float64,int64, cuda
import random
import tensorflow as tf
import glob
import os
import shutil
from distutils.dir_util import copy_tree

quesize = 20
k = 0

@njit
def preprocessing(a):
    px_train = a.reshape(input_size)
    px_train = np.asarray(px_train).astype(np.int32)

    return px_train


class WiSARDrp:                                                     
    
    def __init__(self,input_size,no_of_rand_pix_selec,nodes,ram_address_count,dis_number):
        self.input_size = input_size
        self.no_of_rand_pix_selec = no_of_rand_pix_selec
        self.nodes = nodes
        self.ram_address_count = ram_address_count
        self.dis_number = dis_number 


    def discriminator(self):
        discriminator = []
        accumulated_pos = []
        my_list = list(range(0,self.input_size))
        for i in range(self.dis_number):  #10
            ram = []
            random.shuffle(my_list)
            for j in range((int)((self.nodes))): #98    
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


@njit(parallel = True)
def test_with_bleaching(d,pos,x_test,y_test):
    right = 0
    wrong = 0
    images = x_test
    lable = y_test
    b=1
        
    image = images
    actual_lable = lable
            
    ix = actual_lable       
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
            if prt[0] == address_of_that_ram and prt[1]>=b:           #have to change this
                sum_of_ram_output += 1
            
            
    if sum_of_ram_output/nodes >= 0.75:
        right += 1
    else:
        wrong += 1
        
        
    return right,wrong                
                
                

@njit
def IsFoundinStabilityQueue(stabilityqueue1,i,j,pixel):
    stabilityqueue2 = stabilityqueue1
    c = np.zeros((quesize,3),dtype=np.int32) 
    b = np.zeros((4,3),dtype=np.int32)
    for queueIndex in range(quesize):
        queuevalue = np.reshape(stabilityqueue2[i,j,queueIndex,:],(3,1))
        if queuevalue[0][0] == pixel[0][0] and  queuevalue[1][0] == pixel[1][0] and  queuevalue[2][0] == pixel[2][0] :
            b = np.roll(stabilityqueue2[i,j,:queueIndex+1,:],3) 
            c[:queueIndex+1] = b
            c[queueIndex+2:] = stabilityqueue2[i,j,queueIndex+2:,:]
            stabilityqueue2[i,j,:,:] = c
            return stabilityqueue2,True
    #print(stabilityqueue2.shape)
    return stabilityqueue2,False    
    
    
    
@njit
def bewis(frame,w,ID,stabilityqueue,out,index,d,pos):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            r,g,b = frame[i,j,:]
            r = round((r/256)*128)
            g = round((g/256)*128)
            b = round((b/256)*128)
            #create binary endoder for each pixel
            a = np.zeros((128,3))
            a[0:r,0] = 1  
            a[0:g,1] = 1
            a[0:b,2] = 1
            #check if pixel color is in front of stability queue
            front = np.reshape(stabilityqueue[i,j,0,:],(3,1)) 
            pixel = np.reshape(frame[i,j,:],(3,1))
            pixel_no = i*10+j
            if front[0][0] == pixel[0][0] and  front[1][0] == pixel[1][0] and  front[2][0] == pixel[2][0] :
                w[i,j] += ID[i,j]
                ID[i,j] += 1
            if front[0][0] != pixel[0][0] or  front[1][0] != pixel[1][0] or  front[2][0] != pixel[2][0] :
                stabilityqueue,found = IsFoundinStabilityQueue(stabilityqueue,i,j,pixel)
                if found:
                    ID[i,j] = 1
                else:
                    stabilityqueue[i,j,:,:] = np.roll(stabilityqueue[i,j,:,:],1)
                    stabilityqueue[i,j,0,:]= np.reshape(pixel,(3,))
                    w[i,j] = 0
                    ID[i,j] = 1

            if w[i,j] > k:
                pixel_train = preprocessing(a)
                B = 50    
                images = pixel_train
                lable = pixel_no   
                image = images
                num = lable
                all_ram_of_selected_discriminator = d[num]
                t_ratina = pos[(int)(nodes*num):(int)(nodes*num+nodes)]
            
                for inode in range((int)(nodes)):
                    part = all_ram_of_selected_discriminator[(ram_address_count*inode):(ram_address_count*inode+ram_address_count)]
                    ratina_for_one_ram = t_ratina[inode]
                          
                    n = []                                                                
                    for ix in range(len(ratina_for_one_ram)):
                        pix = ratina_for_one_ram[ix]
                        if image[(pix-1)]>=1:
                            n.append(1)    
                        else:
                            n.append(0)
                
                    num = 0
                    for ino in range(no_of_rand_pix_selec):
                        num = (n[ino])*(10**((no_of_rand_pix_selec-1)-ino)) + num
            
                    address_of_that_ram = (int)(num)
                    for key in range(ram_address_count):
                        index = part[key]
                        if index[0] == address_of_that_ram and index[1] < B:
                            index[1] += 1
                        else:
                            if index[1] != 0 and index[1] != 0:
                                index[1] -= 1
                
            #reconstruct colors from neurons and write it on BG image
            pixel_test = preprocessing(a)
            right,wrong = test_with_bleaching(d,acc_pos,pixel_test,pixel_no)
            if right:
                out[i,j] = 0.
            if wrong:
                out[i,j] = 1.



def UseFrames(framesInputPath,framesOutputPath,files,height=240,width=250):
	stabilityqueue = np.zeros((height,width,quesize,3),dtype=np.int32)   #240,256  #480,640 for bird video, 720,1280 for E_IP1231_Day_1
	w = np.zeros((height,width),dtype=np.int32)
	ID = np.zeros((height,width),dtype=np.int32)
	####cap = cv2.VideoCapture('dice.avi')
	out = np.ones((height,width),dtype=np.float64)
	index = np.zeros((2,),dtype=np.int32)
	n = 0
	no_of_frames = 0
	start = time.time()
	for f in files:
		##print(f)
		#while(True):
		#for file in glob.glob('/home/iss/project/BEWiS/dataset2014/dataset/baseline/highway/input/*jpg'):
		#frame = cv2.imread(file)
		####ret, frame = cap.read()
		frame = cv2.imread(framesInputPath+'/'+f)
		#cv2.imshow('input',frame)
		#width = 250
		#height = 240
		dim = (width, height) 
		# resize image
		resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 
		#Display the resulting frame
		#cv2.imshow('frame',resized)        
		bewis(resized,w,ID,stabilityqueue,out,index,d,acc_pos)
		no_of_frames += 1    
		kernel = np.ones((5,5),np.uint8)
		erosion = cv2.erode(out,kernel,iterations = 1)
		dilation = cv2.dilate(erosion,kernel,iterations = 1)
		#cv2.imshow('out',dilation)
		outpath = framesOutputPath+'/'+'out'+str(no_of_frames)+'.jpg'
		#print(outpath,dilation.shape)
		#dilation.dtype='uint8'
		cv2.imwrite(outpath, cv2.resize(dilation.astype('uint8') * 255, (width,height), interpolation = cv2.INTER_AREA)    )
		#cv2.imwrite(os.path.join(framesOutputPath,'/','out',str(no_of_frames),'.jpg'),dilation)
		   
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	end = time.time()
	tm = end - start
    
	fps = no_of_frames/tm
	print(fps)
	####cap.release()
	cv2.destroyAllWindows()


def LocateInputFolders(rootPath,height=240,width=250):
	dest = 'input'
	for (root,dirs,files) in os.walk(rootPath, topdown=True): 
		l = root.split('/')
		end = l[len(l)-1]
		if(end == dest):
			#print(files)
			files.sort()
			#print(files)
			print ('Root <',root,'>')
			l1 = root.split('/dataset/')
			#print('split -> ',l1)
			temp = l1[0]+'/results/'+l1[1]
			output = temp.split('/input')[0]
			#print('Output ->',output.split('/input')[0])
			UseFrames(root,output,files,height,width)


height= 240		
width= 250

input_size = 128*3
no_of_rand_pix_selec = 2**(2)     ## ** (must) no_of_rand_pix_selec = 2^(n) where n is 0,1,2...
nodes = int(input_size/no_of_rand_pix_selec)    
ram_address_count = 2**(no_of_rand_pix_selec)
dis_number = height*width #240*250
w = WiSARDrp(input_size,no_of_rand_pix_selec,nodes,ram_address_count,dis_number)
d, acc_pos = w.discriminator()
print(d.shape)

LocateInputFolders('/home/iss/project/BEWiS/dataset2014',height,width)











