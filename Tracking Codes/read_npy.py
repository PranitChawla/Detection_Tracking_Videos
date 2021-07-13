import numpy as np 
import cv2
import os 
from PIL import Image 
import PIL 
  
# creating a image object (main image) 
# im1 = Image.open(r"C:\Users\System-Pc\Desktop\flower1.jpg") 
  
# # save a image using extension
# im1 = im1.save("geeks.jpg")

list1 = os.listdir('male_dancer_clipped')
final_path = 'male_dancer_images'


for item in list1:
	image_path = os.path.join('male_dancer_clipped',item)
	item = item.split('_')[-1]
	item = item.split('.')[0]
	item = str(item) + '.png'
	save_path = os.path.join(final_path,item)
	print (save_path)
	im1 = Image.open(image_path)
	im1 = im1.save(save_path)

