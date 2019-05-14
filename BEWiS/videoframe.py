



import numpy as np
import cv2
import itertools

cap = cv2.VideoCapture(0)

quesize = 10
stabilityqueue = np.zeros((480,640,quesize,3))
w = np.zeros((480,640))
i = np.zeros((480,640),dtype = np.int32)
queueFront = np.zeros((480,640))
#print(stabilityqueue[:,:,0,:].shape)


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	frontOfStabilityQueue = np.array(frame == stabilityqueue[:,:,queueFront[:,:],:])
	frontOfStabilityQueue = np.all(frontOfStabilityQueue,axis=2)
	#print(a.shape,a)
	match = np.where(frontOfStabilityQueue==True)
	print(match)	
	
	

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
