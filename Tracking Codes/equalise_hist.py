import cv2
import numpy as np 
import matplotlib.pyplot as plt 

left_points = list()
right_points = list()
left_medians  = list()
right_medians = list()


def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    print len(channels)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img


def compute_medians (left_points, right_points):
	left_points = left_points[len(left_points)-5:len(left_points)]
	right_points = right_points[len(right_points)-5:len(right_points)]





for i in range (1,1000):

	# path = "/home/pranit/DIP_Lab/2_22/Experiment-2/equalised.png"
	path = 'training_dataset/'+'L2V2D4R1_'+str(i)+'.png'
	# print (path)
	img = cv2.imread(path,0)
	img3 = cv2.imread(path,0)
	# eq_input = cv2.imread()
	dst = cv2.equalizeHist(img)
	img4 = cv2.imread(path)	
	track_img = cv2.imread(path)
	eq_image = hisEqulColor(track_img)
	# cv2.imshow("equalised")
	kernel = np.ones((5,5),np.uint8)
	# img = img[360:480,250:400]
	# img2 = img

	imgYCC = cv2.cvtColor(track_img, cv2.COLOR_BGR2YCR_CB)
	cv2.imshow("orig",track_img)
	cv2.imshow("equal",eq_image)
	# cv2.imshow('dst',img)
	# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# img = dst

	ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

	thr = 255-thr

	# thr= cv2.dilate(thr,kernel,iterations = 1)
	img3 = thr/255*img3

	for k in range (img.shape[0]):
		for l in range (img.shape[1]):
			# Y = imgYCC[i][j][0]
			# cr = imgYCC[i][j][1]
			# cb = imgYCC[i][j][2]
			# h = hsv[i][j][0]
			# s = hsv[i][j][1]
			# v = hsv[i][j][2]
			# print (h,s,v)
			# if h<50:
			# 	img3[i][j] = 255
			# else:
			# 	img3[i][j] = 0

			# if 
			r = imgYCC[k][l][0]
			g = imgYCC[k][l][1]
			b = imgYCC[k][l][2]

			# if l>=85 and l<=105 and 
			# if l==90 and k==115:
			# 	print (l,k,r,g,b)

			# if r >=60 and r<=80 and g >= 140 and g <= 160 and b>=120 and b<=140:
			# 	img3[k][l] = 255

			if img3[k][l]>=50 and img3[k][l] <= 90:
			# # if cr<=1.5862*cb+20 and cr>=0.3448*cb+76.2069 and cr<=-4.5652*cb+234.5652 and cr>=-1.15*cb+301.75 and cr<=-2.2857*cb+432.85:
				img3[k][l] = 255
			else:
				img3[k][l] = 0


	# # find the keypoints with ORB
	# kp = orb.detect(img)

	# print (queryKeypoints)
	# compute the descriptors with ORB
	# kp, des = orb.compute(img, kp)
	cv2.imshow("check",img3)
	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(img3,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,kernel,iterations = 1)
	cv2.imshow("win3",dilation)
	# cv2.waitKey(30)
	# continue
	rows,cols = dilation.shape


	white_mask = np.ones((rows,cols//2))
	black_mask = np.zeros((rows,cols//2))

	# print (white_mask.shape,black_mask.shape,dilation.shape)

	left_mask = np.zeros((dilation.shape))
	left_mask[:,:cols//2] = white_mask
	left_mask[:,cols//2:] = black_mask

	right_mask = np.zeros((dilation.shape))
	right_mask[:,:cols//2] = black_mask
	right_mask[:,cols//2:] = white_mask

	# print (white_mask.shape,left_mask.shape)
	print (left_mask.shape)

	print (dilation.shape)

	dilation_left = cv2.bitwise_and(dilation, left_mask.astype('uint8')

		)
	dilation_right = cv2.bitwise_and(dilation, right_mask.astype('uint8')

		)

	# dilation_left = dilation*left_mask


	# print (left_mask.shape)

	# print (dilation_left)

	contours_left, hierarchy_left = cv2.findContours(dilation_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_right, hierarchy_right = cv2.findContours(dilation_right, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours_refined_left = list()
	contours_refined_right = list()
	# for contour in contours:
	# 	# print (cv2.contourArea(contour))
	# 	if cv2.contourArea(contour) > 100:
	# 		contours_refined.append(contour)

	contours_refined_left = sorted(contours_left,key=lambda x:cv2.contourArea(x), reverse=True)

	contours_refined_left = contours_refined_left[:1]

	contours_refined_right = sorted(contours_right,key=lambda x:cv2.contourArea(x), reverse=True)

	contours_refined_right = contours_refined_right[:1]


	# print (len(contours_refined))
	cv2.drawContours(img4, contours_refined_left, -1, (0,255,0), 3)
	cv2.drawContours(img4, contours_refined_right, -1, (0,255,0), 3)

	M_left = cv2.moments(contours_refined_left[0])
	left_x = int(M_left["m10"] / M_left["m00"])
	left_y = int(M_left["m01"] / M_left["m00"])

	M_right = cv2.moments(contours_refined_right[0])
	right_x = int(M_right["m10"] / M_right["m00"])
	right_y = int(M_right["m01"] / M_right["m00"])


	# print (cX1,cY1, cX2, cY2)


	# if cX1 > cX2:
	# 	left = contours_refined[1]
	# 	right = contours_refined[0]

	# else:
	# 	left = contours_refined[0]
	# 	right = contours_refined[1]


	# M_left = cv2.moments(left)
	# left_x = int(M_left["m10"] / M_left["m00"])
	# left_y = int(M_left["m01"] / M_left["m00"])

	# M_right = cv2.moments(right)
	# right_x = int(M_right["m10"] / M_right["m00"])
	# right_y = int(M_right["m01"] / M_right["m00"])	

	if i==1:
		x_left_center_permanent = left_x
		x_right_center_permanent = right_x
		y_max_left = left_y
		y_max_right = right_y





	right_y = min(right_y,y_max_right)
	left_y = min(left_y,y_max_left)


	if abs(right_y-y_max_right) > 50:
		right_y = right_points[len(right_points)-2][1]


	if abs(left_y-y_max_left) > 50:
		left_y = left_points[len(left_points)-2][1]

	print (left_x,left_y, right_x, right_y)
	left_points.append((x_left_center_permanent,left_y))
	right_points.append((x_right_center_permanent,right_y))

	# if i>5:


	thickness = 5
	if i>10:
		cv2.line(track_img, left_points[i-2], left_points[i-1], (0, 0, 255), thickness)
		# cv2.line(track_img, left_points[i-2], left_points[i-1], (0, 0, 255), thickness)
		cv2.line(track_img, right_points[i-2], right_points[i-1], (0, 0, 255), thickness)
	# # draw only keypoints location,not size and orientation
	cv2.imshow("win2",img4)
	cv2.imshow("win",track_img)
	# cv2.imshow()
	# print (imgYCC[1])
	# break
	cv2.waitKey(30)
