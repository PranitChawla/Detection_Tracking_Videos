import argparse
import time
import os
import torch
import torch.nn as nn
from torchvision import transforms
from data_loader import get_loader
import torch.nn.functional as F
import numpy as np
from models import C3D
from torch.autograd import Variable
# Device configuration
torch.cuda.set_device(0)
IMAGE_ROOT = "training_dataset"
LABELS = "intervals.json"


criterion = nn.BCELoss()


def train(args):
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    transform_dev = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])


#     Build data loader
    data_loader = get_loader(IMAGE_ROOT,
                             LABELS,
                             transform,
                             args.batch_size, shuffle=True, return_target=True, num_workers=args.num_workers)

    # for i, (candidate_labels,candidate_images,meta) in enumerate(data_loader):
    #     print (meta)
    def logging(s, print_=True, log_=True):
        if print_:
            print(s)

    logging(str(args))
    total_step = len(data_loader)
    image_encoder_obj = C3D().cuda()
    params = image_encoder_obj.get_trainable_parameters()

    current_lr = args.learning_rate
    optimizer= torch.optim.Adam(params, lr=current_lr)

    for epoch in range(100):
#         if (epoch>=10):
#             caption_encoder.embed.weight.requires_grad = True

        for i, (candidate_labels,candidate_video) in enumerate(data_loader):           
        
            candidate_video = candidate_video.cuda()
            print (candidate_video.shape)
            candidate_ft = image_encoder_obj.forward(candidate_video)
            print (candidate_ft.shape)
            candidate_ft = candidate_ft.cuda()
            labels = Variable(candidate_labels,requires_grad=False).cuda()
            # print (candidate_ft.shape,labels.shape)
            loss = criterion(candidate_ft,labels)
            loss.backward()                
            optimizer.step()
            if i % args.log_step == 0:
                logging(
                    '| epoch {:3d} | step {:6d}/{:6d} | lr {:06.6f} | train loss {:8.3f}'.format(epoch, i, total_step,
                                                                                                 current_lr,
                                                                                                 loss.item()))
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='models',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')

    parser.add_argument('--data_set', type=str, default='dress')
    parser.add_argument('--log_step', type=int, default=20,
                        help='step size for printing log info')
    parser.add_argument('--patient', type=int, default=3,
                        help='patient for reducing learning rate')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512,
                        help='dimension of word embedding vectors')
    # Learning parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()

    train(args)

