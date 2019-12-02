import numpy as np
import cv2
import itertools
import time
import random
import tensorflow as tf
import glob
import os
import shutil
from distutils.dir_util import copy_tree
import subprocess
import sys
from skimage.measure import compare_ssim
from sklearn.metrics import confusion_matrix

#import Stats

call = subprocess.call


def deleteIfExists(path):
	if os.path.exists(path):
		os.remove(path)

def compareWithGroungtruth(statFilePath,videoPath, binaryPath):
	"""Compare your binaries with the groundtruth and return the confusion matrix"""
	#statFilePath = os.path.join(videoPath, 'stats.txt')
	#deleteIfExists(statFilePath)

	retcode = call([os.path.join('/home/iss/project/BEWiS/PythonCode2014/python_c++/exe', 'Comparator.exe'),videoPath, binaryPath], shell=True)
	return retcode,readCMFile(statFilePath)

def readCMFile(filePath):
	"""Read the file, so we can compute stats for video, category and overall."""
	if not os.path.exists(filePath):
		print("The file " + filePath + " doesn't exist.\nIt means there was an error calling the comparator.")
		return 0
	
	with open(filePath) as f:
		for line in f.readlines():
			if line.startswith('cm:'):
				numbers = line.split()[1:]
				return [int(nb) for nb in numbers[:5]]

						
		
def CompareOutputs(rootPath,width,height):
	dest = 'groundtruth'
	for (root,dirs,files) in os.walk(rootPath, topdown=True): 
		l = root.split('/')
		end = l[len(l)-1]
		if(end == dest):
			#print(files)
			files.sort()
			#print(files)
			#print ('Root <',root,'>')
			l1 = root.split('/dataset/')
			#print('split -> ',l1)
			temp = l1[0]+'/results/'+l1[1]
			output = temp.split('/groundtruth')[0]
			ROI = cv2.imread((root.split('/groundtruth'))[0] +'/ROI.bmp', cv2.IMREAD_GRAYSCALE)
			ROI = cv2.resize(ROI.astype('uint8') * 255, (width,height) , interpolation = cv2.INTER_AREA)
			#print('ROI ---', ROI)
			#print(gtp)
			for (root1,dirs1,files1) in os.walk(output, topdown=True):
				files1.sort()	
				#print( root1, len(files1), len(files) )	
			for i in range(0, len(files1)):	
				#print(output+'/'+files1[i],root+'/'+files[i]," :: ")
				outputROI = cv2.imread(output+'/'+files1[i], cv2.IMREAD_GRAYSCALE) 
				outputROI = cv2.resize(outputROI.astype('uint8') * 255, (width,height) , interpolation = cv2.INTER_AREA)
				gtROI = cv2.imread(root +'/'+files[i], cv2.IMREAD_GRAYSCALE)
				gtROI = cv2.resize(gtROI.astype('uint8') * 255, (width,height) , interpolation = cv2.INTER_AREA)
				#print( ROI.shape, outputROI.shape, gtROI.shape )
				
				outputROI = cv2.bitwise_and(outputROI,ROI)
				gtROI = cv2.bitwise_and(gtROI,ROI)
				
				#(score, diff) = compare_ssim(outputROI, gtROI, full=True)
				#diff = (diff * 255).astype("uint8")
				#print("SSIM: {}".format(score))
				tn, fp, fn, tp = confusion_matrix(outputROI.flatten(), gtROI.flatten()).ravel()			
				#reult2 = confusion_matrix(outputROI, gtROI)
				print(tn,'    ', fp, '    ', fn,'    ', tp)
				#retcode, C = compareWithGroungtruth(output+'/'+'stats.txt',outputROI,outputROI)
				#print(retcode)

	

height= 240		
width= 250

CompareOutputs('/home/iss/project/BEWiS/dataset2014',height,width)

