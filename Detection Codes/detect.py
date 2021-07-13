import os
import shutil
from PIL import Image
import numpy as np
import torch
path = "training_dataset"
flow_path = "selflow_outputs"
from crd_model import C3D, combiner

from torchvision import transforms
list1 = os.listdir(path)
list1 = [x for x in list1 if "L2V1D1R1" in x]

transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
# print (list1)
image_encoder_obj = C3D().cuda()
flow_model = C3D().cuda()
combine = combiner().cuda()

image_encoder_obj.load_state_dict(torch.load("3d_conv_model.pth"))
flow_model.load_state_dict(torch.load("flow_model.pth"))
combine.load_state_dict(torch.load("combine_model.pth"))
list_temp = []
combine.eval()
flow_model.eval()
image_encoder_obj.eval()
# stack2 = np.zeros((480,640,3))
# final = np.stack((stacked_images,stack2),0)
# print (final.shape)
for i in range (6,len(list1)):
	candidate_video = []
	flow_video = []
	for j in range (i-6,i):
		img_name = "L2V1D1R1" + "_" + str(i) + '.png'
		candidate_image = Image.open(path+"/"+img_name)
		candidate_image = transform(candidate_image)
		candidate_video.append(candidate_image)
		try:
			flow_image_fw = Image.open(flow_path+"/"+"flow_fw_color_"+img_name)
			flow_image_bw = Image.open(flow_path+"/"+"flow_bw_color_"+img_name)
			flow_image_fw = transform(flow_image_fw)
			flow_image_bw = transform(flow_image_bw)
			flow_image = (flow_fw_img + flow_bw_img)/2.0
		except:
			flow_image = torch.ones(3,160,150)
		flow_video.append(flow_image)
		# print (candidate_image)
	candidate_video = torch.stack(candidate_video)
	flow_video = torch.stack(flow_video)
	candidate_video = candidate_video.permute(1,0,2,3)
	flow_video = flow_video.permute(1,0,2,3)
	candidate_video = candidate_video.cuda()
	flow_video = flow_video.cuda()
	candidate_video = candidate_video.unsqueeze(0)
	flow_video = flow_video.unsqueeze(0)
	candidate_ft = image_encoder_obj(candidate_video)
	flow_ft = flow_model(flow_video)
	candidate_ft = candidate_ft.unsqueeze(0)
	flow_ft = flow_ft.unsqueeze(0)
	final_label = combine((candidate_ft,flow_ft))
	final_label = final_label.squeeze()
	if float(final_label)>=0.9:
		print (final_label,i)


	


