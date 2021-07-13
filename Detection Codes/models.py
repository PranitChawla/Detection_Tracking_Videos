import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import torchvision
from torch.utils import model_zoo


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


"""
Pulled from https://github.com/kenshohara/3D-ResNets-PyTorch
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 hidden_layer= 2048,
                 out_dim = 128,
                 projection_head=True):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0] #64
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        #self.attention = SpatialAttention(in_channels=(block_inplanes[3] * block.expansion))

        # Spatio temporal attention
        self.channel_temporal_attention1 = ChannelTemporalAttention(in_channels = block_inplanes[0] * block.expansion)
        self.channel_temporal_attention2 = ChannelTemporalAttention(in_channels = block_inplanes[1] * block.expansion)
        self.channel_temporal_attention3 = ChannelTemporalAttention(in_channels = block_inplanes[2] * block.expansion)
        self.channel_temporal_attention4 = ChannelTemporalAttention(in_channels = block_inplanes[3] * block.expansion)
        self.spatio_temporal_attention1 = SpatioTemporalAttention(in_channels = block_inplanes[0] * block.expansion)
        self.spatio_temporal_attention2 = SpatioTemporalAttention(in_channels = block_inplanes[1] * block.expansion)
        self.spatio_temporal_attention3 = SpatioTemporalAttention(in_channels = block_inplanes[2] * block.expansion)
        self.spatio_temporal_attention4 = SpatioTemporalAttention(in_channels = block_inplanes[3] * block.expansion)

        # TODO: don't use fixed numbers
        #self.attention_mask_avgpool = nn.AdaptiveAvgPool3d((2, 8, 8))
        #self.conv_attention_feature_refinement = nn.Conv3d(in_channels=4,
        #                                           out_channels = block_inplanes[3] * block.expansion,
        #                                           kernel_size=(1, 1, 1),
        #                                           stride=(1, 1, 1),
        #                                           padding=(0, 0, 0))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection_head = projection_head
        if projection_head:
            print('==> setting up non-linear project heads')
            self.fc1 = nn.Linear(block_inplanes[3] * block.expansion, hidden_layer)
            self.bn_proj = m = nn.BatchNorm1d(hidden_layer)
            self.fc2 = nn.Linear(hidden_layer, out_dim)
        else:
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, hidden_layer)


        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('forwarding in resnet module')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.channel_temporal_attention1(x)
        x, M_s1 = self.spatio_temporal_attention1(x)

        x = self.layer2(x)
        x = self.channel_temporal_attention2(x)
        x, M_s2 = self.spatio_temporal_attention2(x)

        x = self.layer3(x)
        x = self.channel_temporal_attention3(x)
        x, M_s3 = self.spatio_temporal_attention3(x)

        x = self.layer4(x)
        x = self.channel_temporal_attention4(x)
        x, M_s4 = self.spatio_temporal_attention4(x)

        # attention guided feature refinement
        #M_s1 = self.attention_mask_avgpool(M_s1)
        #M_s2 = self.attention_mask_avgpool(M_s2)
        #M_s3 = self.attention_mask_avgpool(M_s3)
        #M_ms = torch.cat((M_s1, M_s2, M_s3, M_s4), dim=1)  # 4xTxHxW
        #M_ms = self.conv_attention_feature_refinement(M_ms)
        #M_ms = self.relu(M_ms)
        #x = x + M_ms

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        #add projection head
        if self.projection_head:
            x = self.fc1(x)
            x = self.bn_proj(x)
            # print('after fc1', x.size())
            x = self.relu(x)
            # print('after relu', x.size())
            x = self.fc2(x)
            # print('after fc2', x.size())

        return x


## channel-temporal + spatio-temporal attention

class ChannelTemporalAttention(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.spatial_avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.spatial_maxpool = nn.AdaptiveMaxPool3d((None, 1, 1))

        # MLP - channel reasoning
        r = 4  # reduction
        hidden_layer = in_channels // r
        self.fc1 = nn.Linear(in_channels, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, in_channels)
        self.sigmoid = nn.Sigmoid()

        # CNN with two 1d convs - temporal reasoning
        # TODO channels should be separate?

        k=3
        self.conv1d_1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=k,
                                stride=1,
                                padding=k//2)
        self.conv1d_2 = nn.Conv1d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=k,
                                stride=1,
                                padding=k//2)

    def forward(self, x):
        x = x.transpose(1,2)  # TxCxHxW

        # Channel descriptors
        d_avgc = self.spatial_avgpool(x)  # TxCx1x1
        d_maxc = self.spatial_maxpool(x)  # TxCx1x1
        d_avgc = d_avgc.squeeze(4).squeeze(3)  # TxC
        d_maxc = d_maxc.squeeze(4).squeeze(3)  # TxC

        # MLP, to produce two channel frame attention descriptors
        d_avgc = self.fc2(self.fc1(d_avgc))  # TxC
        d_maxc = self.fc2(self.fc1(d_maxc))  # TxC

        # Elem-wise sum of channel frame descriptors + sigmoid
        M_c = self.sigmoid(d_avgc + d_maxc)  # TxC
        M_c = M_c.transpose(1,2)  # CxT

        # CNN with two 1D convs
        M_c = self.conv1d_1(M_c)  # CxT
        M_c = self.conv1d_2(M_c)  # CxT
        M_c = self.sigmoid(M_c)

        # Apply channel-temporal mask
        M_c = M_c.unsqueeze(-1).unsqueeze(-1)  #CxTx1x1
        x = x.transpose(1,2)  # CxTxHxW
        x = M_c * x

        return x


class SpatioTemporalAttention(torch.nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.channel_avgpool = nn.AdaptiveAvgPool3d((1, None, None))
        self.channel_maxpool = nn.AdaptiveMaxPool3d((1, None, None))

        # CNN with one 2d conv - spatial reasoning
        k=7
        self.conv2d = nn.Conv3d(in_channels=2,
                                out_channels=1,
                                kernel_size=(1,k,k),
                                stride=(1,1,1),
                                padding=(0,k//2,k//2))
        self.sigmoid = nn.Sigmoid()

        # CNN with two 3d convs - temporal reasoning
        self.conv3d_1 = nn.Conv3d(in_channels=1,
                               out_channels=1,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(in_channels=1,
                               out_channels=1,
                               kernel_size=(3, 3, 3),
                               stride=(1, 1, 1),
                               padding=(1, 1, 1))

    def forward(self, x):

        x = x.transpose(1,2)  # TxCxHxW

        # Spatial feature maps
        d_avgs = self.channel_avgpool(x)  #Tx1xHxW
        d_maxs = self.channel_avgpool(x)  #Tx1xHxW

        # CNN with one 2d conv
        M_s = torch.cat((d_avgs, d_maxs), dim=2)  #Tx2xHxW
        M_s = M_s.transpose(1,2)  # 2xTxHxW
        M_s = self.conv2d(M_s)  # 1xTxHxW
        M_s = self.sigmoid(M_s)

        # CNN with two 3d convs
        M_s = self.conv3d_1(M_s)
        M_s = self.conv3d_2(M_s)
        M_s = self.sigmoid(M_s)  # 1xTxHxW

        # Apply spatio-temporal mask
        x = x.transpose(1,2)
        x = M_s * x

        return x, M_s

##


class SpatialAttention(torch.nn.Module):

    def __init__(self, in_channels=512):
        super().__init__()

        conv_t_size = 1
        conv_t_stride = 1
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=1024,
                               kernel_size=(conv_t_size, 3, 3),
                               stride=(conv_t_stride, 1, 1),
                               padding=(conv_t_size // 2, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(1024)

        self.conv2 = nn.Conv3d(in_channels=1024,
                               out_channels=512,
                               kernel_size=(conv_t_size, 3, 3),
                               stride=(conv_t_stride, 1, 1),
                               padding=(conv_t_size // 2, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(512)

        self.conv3 = nn.Conv3d(in_channels=512,
                               out_channels=1,
                               kernel_size=(conv_t_size, 3, 3),
                               stride=(conv_t_stride, 1, 1),
                               padding=(conv_t_size // 2, 1, 1),
                               bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.conv1(x)
        mask = self.bn1(mask)
        mask = self.relu(mask)

        mask = self.conv2(mask)
        mask = self.bn2(mask)
        mask = self.relu(mask)

        mask = self.conv3(mask)
        mask = self.sigmoid(mask)

        x = x * mask
        return x


def generate_model(model_depth, **kwargs):
    def get_inplanes():
        return [64, 128, 256, 512]

    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model




if __name__ == '__main__':

    model_depth=200
    n_classes=1039
    n_input_channels=3
    resnet_shortcut = 'B'
    conv1_t_size = 7 #kernel size in t dim of conv1
    conv1_t_stride = 1 #stride in t dim of conv1
    no_max_pool = False #max pooling after conv1 is removed
    resnet_widen_factor = 1 #number of feature maps of resnet is multiplied by this value


    model=generate_model(model_depth=model_depth, n_classes=n_classes,
                        n_input_channels=n_input_channels, shortcut_type=resnet_shortcut,
                        conv1_t_size=conv1_t_size,
                        conv1_t_stride=conv1_t_stride,
                        no_max_pool=no_max_pool,
                        widen_factor=resnet_widen_factor)

    print(model)