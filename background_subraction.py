import numpy as np
import imutils
import cv2 
import os
# from matplotlib import pyplot as plt


def get_median (image):
	list_points = []
	for i in range (image.shape[0]):
		for j in range (image.shape[1]):
			if image[i][j]:
				list_points.append(i);
	list_points.sort()
	# print (len(list_points))
	if list_points!= []:
		return list_points[len(list_points)//2]
	else:
		return -1

path = "data/AVI"
list1 = os.listdir(path)



background = cv2.imread('data/AVI/0.png',1)
background = background[320:480,250:400]
background_gray = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)
backSub = cv2.createBackgroundSubtractorKNN()
# background_gray = backSub.apply(background_gray)


for i in range (1,len(list1)):
	img = cv2.imread('data/AVI/'+str(i)+'.png',1)
	img = img[320:480,250:400]
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	fgMask = backSub.apply(gray)


	# kernel = np.ones((3, 3), np.uint8) 
	# fgMask = cv2.erode(fgMask, kernel)  
	# fgMask = cv2.dilate(fgMask,kernel)
	# img4 = cv2.absdiff(gray,background_gray)
	# img4 = cv2.erode(img4, kernel)  
	# img4 = cv2.dilate(img4,kernel,iterations = 1)
	# Initiate STAR detector
	
	# h, w = img4.shape[:2]
	

	# img3 = cv2.absdiff(fgMask,background_gray)
	# cv2.imshow('Frame', img3)
	img3_left = fgMask[:,:fgMask.shape[1]//2-10]
	# # print (img3_left.shape)
	img3_right = fgMask[:,fgMask.shape[1]//2+10:fgMask.shape[1]]

	median_left = get_median(img3_left)
	median_right = get_median(img3_right)



	


	cv2.imshow("mask",img3_left)
	cv2.imshow("mask1",img3_right)

	cv2.imwrite("tracking_results/"+"left"+str(i)+".jpg",img3_left)
	cv2.imwrite("tracking_results/"+"right"+str(i)+".jpg",img3_right)
	# cv2.imshow("back",background_gray)

	# print (np.max(img3_left),np.max(img3_right))

	# mask_left = (img3_left>255)
	# mask_right = (img3_right==255)


	# sum_left = np.sum(mask_left)
	# sum_right = np.sum(mask_right)
	# print (sum_left,sum_right)
	# if (sum_left>100):
	# 	print ("left up, ",i)
	# if (sum_right>100):
	# 	print ("right up, ",i)
	# background_gray = gray
	# cv2.imshow("check",img3)

	
	cv2.waitKey(30)
	if median_left-median_right>35:
		print (i,"left")
	elif median_left-median_right<-35:
		print(i,'right')
	else:
		print (i,'ground')
	# print (i,median_left-median_right)
