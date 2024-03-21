
# as simple as possible,

import math
from collections import OrderedDict
from typing import Optional
import logging

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
import numpy as np
import torch.nn as nn
import torch

from torch import Tensor
from typing import Type
# inspired by https://gist.github.com/knsong/ac3b8205d86098d14754d02f908942ea
# https://github.com/robmarkcole/resent18-from-scratch/blob/main/resnet18.py
# class BasicBlock(nn.Module):
#     """Basic Block for ResNet18 and ResNet34"""
#     def __init__(
#         self,in_channels,out_channels,stride):
#         super(BasicBlock, self).__init__()
#         # Multiplicative factor for the subsequent conv2d layer's output channels.
#         # It is 1 for ResNet18 and ResNet34.
#         if stride != 1:
#             self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # here stride should be 1

#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = 1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#     def forward(self, x):
#         identity = x # store copy of input tensor
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity # add input tensor to output tensor - residual connection
#         out = self.relu(out)
#         return out

# class ResNet18(nn.Module):
#     def __init__(self,audio_channels,num_layers):
#         super(ResNet18, self).__init__()
#         layers = [2, 2, 2, 2]
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(
#             in_channels=audio_channels,
#             out_channels=self.in_channels,
#             kernel_size=7,
#             stride=2,
#             padding=3,
#             bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(self.in_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512*self.expansion, num_classes)

#     def _make_layer(self,out_channels,blocks,stride):
#         if stride != 1:
#             """
#             This should pass from `layer2` to `layer4` or 
#             when building ResNets50 and above. Section 3.3 of the paper
#             Deep Residual Learning for Image Recognition
#             (https://arxiv.org/pdf/1512.03385v1.pdf).
#             """
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
#                 nn.BatchNorm2d(out_channels)
#                 )
#         layers = [] # for storing the layers
#         layers.append(
#             block(
#                 self.in_channels, out_channels, stride, self.expansion, downsample
#             )
#         )
#         self.in_channels = out_channels * self.expansion
#         for i in range(1, blocks):
#             layers.append(block(
#                 self.in_channels,
#                 out_channels,
#                 expansion=self.expansion
#             ))
#         return nn.Sequential(*layers)
#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # The spatial dimension of the final layer's feature
#         # map should be (7, 7) for all ResNets.
#         # print('Dimensions of the last convolutional feature map: ', x.shape)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x

# don't wannat change it more

class BasicBlock(nn.Module):
    """Basic Block for ResNet18 and ResNet34"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x # store copy of input tensor
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity # add input tensor to output tensor - residual connection
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        img_channels: int, # either grayscale or RGB images
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = [] # for storing the layers
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # we need the embedding so we stop it here.
        return x

# we need a decoder here
# https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
# https://blog.csdn.net/dcrmg/article/details/84396211
class BasicBlockDec(nn.Module):

    def __init__(self, in_channel, stride=1):
        super().__init__()
        out_channel = in_channel // 2
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.shortcut = nn.Sequential() # question, should we add something here?
        else:
            self.conv1 = nn.convtranspose2d(in_channel, out_channel, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(out_channel)
            self.shortcut = nn.Sequential(
                nn.convtranspose2d(in_channel, out_channel, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(out_channel)
                out = torch.relu(out)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        return out
class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, in_channel=1):
        super().__init__()
        self.in_channel = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = nn.convtranspose2d(64, in_channel, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, out_channel, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_channel, stride)]
        self.in_channel = out_channel
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x)) # the output is 0 - 1
        return x
    
# VAE model 
# https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py