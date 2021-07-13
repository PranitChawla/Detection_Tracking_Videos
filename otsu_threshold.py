
import cv2
import numpy as np 
import matplotlib.pyplot as plt 


orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for i in range (1,1000):
	img1 = cv2.imread('training_dataset/L2V3D4R1_'+str(i)+'.png')
	img = cv2.imread('training_dataset/L2V3D4R1_'+str(i)+'.png',0)  #pass 0 to convert into gray level 
	  #pass 0 to convert into gray level 
	ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
	kernel = np.ones((5,5),np.uint8)
	thr = 255-thr
	# dilation = cv2.dilate(thr,kernel,iterations = 1)
	dilation = thr
	print ("going")
	# img = dilation/255*img

	if i==1:
		curr = dilation
		prev = dilation
	else:
		prev = curr
		curr = dilation

	kp1, des1 = orb.detectAndCompute(prev,None)
	kp2, des2 = orb.detectAndCompute(curr,None)

	matches = bf.match(des1,des2)

	matches = sorted(matches, key = lambda x:x.distance)

	img3 = img1

	img3 = cv2.drawMatches(prev,kp1,curr,kp2,matches[:10], img3, flags=2)

	# print (kp1,kp2)
	# img3 = img1
	# cv2.drawKeypoints(img3,kp2,img3,color=(255,0,0), flags=0)




	# cv2.imshow('win1', dilation)
	cv2.imshow("win2", img3)
	cv2.imshow("win",img)
	cv2.waitKey(0)  
# image = img
# pixel_vals = image.reshape((-1,3)) 
  
# # Convert to float type 
# pixel_vals = np.float32(pixel_vals)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
  
# # then perform k-means clustering wit h number of clusters defined as 3 
# #also random centres are initally chosed for k-means clustering 
# k = 4
# retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
  
# # convert data into 8-bit values 
# centers = np.uint8(centers) 
# segmented_data = centers[labels.flatten()] 
  
# # reshape data into the original image dimensions 
# segmented_image = segmented_data.reshape((image.shape)) 
  
# cv2.imshow("check",segmented_image)
# cv2.waitKey(0)