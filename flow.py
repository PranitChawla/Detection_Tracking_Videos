import cv2 
import numpy as np 
import os
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


# old_frame = cv.imread("data/AVI/0.png")


list1 = os.listdir("data/AVI/")
frame1 = cv2.imread('/home/pranit/btech_project/data/AVI/0.png',1)
frame1 = frame1[360:480,250:400]
for i in range (1,len(list1)):

	frame2 = cv2.imread('/home/pranit/btech_project/data/AVI/'+str(i)+'.png',1)
	frame2 = frame2[360:480,250:400]
	prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	# prvs = prvs[360:480,250:400]
	# next = next[360:480,250:400]
	flow_image = np.zeros_like(frame1);
# print flow_image.shape
	flow_image[...,1] = 255
	# print (prvs.shape,next.shape)
	flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

	# Change here
	horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)
	vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
	horz = horz.astype('uint8')
	vert = vert.astype('uint8')

	flow_image[..., 0] = ang * 180 / np.pi / 2;
	flow_image[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX);
	rgb = cv2.cvtColor(flow_image, cv2.COLOR_HSV2BGR);
	print (rgb.shape)
	mask = horz>250
	mask2 = vert>250
	horz = horz*mask
	vert = vert*mask
	print (horz)
	# Change here too
	cv2.imshow('Horizontal Component', horz)
	cv2.imshow('Vertical Component', vert)
	cv2.imshow('flow_image',rgb)
	cv2.imwrite("tracking_results/"+"flow_"+str(i)+".jpg",rgb)
	cv2.waitKey(30)

	# k = cv2.waitKey(0) & 0xff
	# if k == ord('s'): # Change here

	# cv2.imwrite('flow_outputs/opticalflow_horz_'+str(i)+'.pgm', horz)
	# cv2.imwrite('flow_outputs/opticalflow_vert_'+str(i)+'.pgm', vert)
	# cv2.imwrite('flow_outputs/c_python_'+str(i)+'.jpg', rgb)
	frame1 = frame2





	











# cv2.destroyAllWindows()