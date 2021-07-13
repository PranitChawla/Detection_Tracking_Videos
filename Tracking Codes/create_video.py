import cv2
import numpy as np
import glob

img_array = []
for i in range (11,900):
    filename = 'video2/'+ str(i) + '.jpg'
    print (filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


# print (len(img_array))
# print (img_array)

out = cv2.VideoWriter('project2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15 , size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()