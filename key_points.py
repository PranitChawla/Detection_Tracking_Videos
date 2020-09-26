import numpy as np
import cv2 
import os
# from matplotlib import pyplot as plt

path = "data/AVI"
list1 = os.listdir(path)

for i in range (len(list1)):
	img = cv2.imread('data/AVI/'+str(i)+'.png',1)
	img = img[360:480,250:400]
	img2 = img
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Initiate STAR detector
	orb = cv2.ORB_create() 
	queryKeypoints, queryDescriptors = orb.detectAndCompute(gray,None)
	gray = np.float32(gray)
	# dst = cv2.cornerHarris(gray,2,3,0.04)
	# dst = cv2.dilate(dst,None)
	# img[dst>0.01*dst.max()]=[0,0,255]
	cv2.drawKeypoints(img2,queryKeypoints,img,color=(255,255,255), flags=0)
	cv2.imshow("window",img2)
	cv2.imwrite("tracking_results/"+str(i)+".jpg",img2)
	# cv2.imshow('dst',img)


	# # find the keypoints with ORB
	# kp = orb.detect(img)

	# print (queryKeypoints)
	# compute the descriptors with ORB
	# kp, des = orb.compute(img, kp)

	# # draw only keypoints location,not size and orientation
	
	cv2.waitKey(30)
# minHessian = 400
# detector = cv.xfeatures2d.SURF(hessianThreshold=minHessian)
# keypoints = detector.detect(img)
# img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
# cv.drawKeypoints(src, keypoints, img_keypoints)
# #-- Show detected (drawn) keypoints
# cv.imshow('SURF Keypoints', img_keypoints)
# cv.waitKey()
# cv.SIFT()