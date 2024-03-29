import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import torchvision
from torch.utils import model_zoo



class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs


    def get_trainable_parameters(self):
        return list(self.parameters())

class image_encoder (nn.Module):
	def __init__(self,embed_dim):
		super(image_encoder, self).__init__()
		resnet_50 = models.resnet50(pretrained=True)
		resnet_50.fc = torch.nn.Sequential(torch.nn.Linear(4*embed_dim, embed_dim))
		self.image_encoder = resnet_50
		self.classifer = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(embed_dim, embed_dim//2),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(embed_dim//2, embed_dim//4),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(embed_dim//4, embed_dim//8),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(embed_dim//8, 1),
                            nn.Sigmoid()
                                )
	def forward (self,image):
		image_ft = self.image_encoder(image)
		output_ft = self.classifer(image_ft)
		output_ft = output_ft.squeeze()
		return output_ft

	def get_trainable_parameters(self):
		return (list(self.parameters()))
