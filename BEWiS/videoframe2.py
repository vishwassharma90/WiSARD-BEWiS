


from numba import njit,jit
import numpy as np
import cv2
import itertools

cap = cv2.VideoCapture(0)

quesize = 10
stabilityqueue = np.zeros((480,640,quesize,3),dtype=np.int32)
w = np.ndarray((480,640),dtype=np.int32)
ID = np.ndarray((480,640),dtype=np.int32)
queueLimits = np.zeros((480,640,2),dtype=np.int32)
#print(stabilityqueue[:,:,0,:].shape)

@njit
def IsFoundinStabilityQueue(pixel):
	for queueIndex in range(queueLimits[i,j,0],queueLimits[i,j,1]):
		queuevalue = np.reshape(stabilityqueue[i,j,queueIndex,:],(3,1))
		if queuevalue[0][0] == pixel[0][0] and  queuevalue[1][0] == pixel[1][0] and  queuevalue[2][0] == pixel[2][0] :	
			return True
	return False
	
@njit
def bewis(frame):
	for i in range(frame.shape[0]):
		for j in range(frame.shape[1]):
			r,g,b = frame[i,j,:]
			#create binary endoder for each pixel
			a = np.zeros((256,3))
			a[0:r,0] = 1  
			a[0:g,1] = 1
			a[0:b,2] = 1
			#check if pixel color is in front of stability queue
			front = np.reshape(stabilityqueue[i,j,queueLimits[i,j,0],:],(3,1))  	
			pixel = np.reshape(frame[i,j,:],(3,1))
			if front[0][0] == pixel[0][0] and  front[1][0] == pixel[1][0] and  front[2][0] == pixel[2][0] :	
				w[i,j] = ID[i,j]
				ID[i,j] += 1
			elif IsFoundinStabilityQueue(pixel):
				x=0

						
					
				

	


while(True):
	
	ret, frame = cap.read()
	# Display the resulting frame
	cv2.imshow('frame',frame)
	
	bewis(frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


'''
	print(frame.shape)
	for p in itertools.product(range(frame.shape[0]),range(frame.shape[1])):
		print (p)
		#r,g,b = frame[p[0],p[1],:]
		#a = np.zeros((256,3))
		#a[0:r,0] = 1  
		#a[0:g,1] = 1
		#a[0:b,2] = 1
'''	
