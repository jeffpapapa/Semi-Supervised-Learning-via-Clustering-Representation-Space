# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:16:00 2020

@author: Yen-Chieh, Huang

Building network for MNIST

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # define nn
    def __init__(self, output_dim, cluster_dim):
        super(Net, self).__init__()
        
        ### Encoder
        self.conv1 = nn.Conv2d(1,32,5)
        self.conv_bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv_bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,3)
        self.conv_bn3 = nn.BatchNorm2d(64)
        self.pool_bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128,3)
        self.conv_bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,cluster_dim,1)
        self.conv_bn5 = nn.BatchNorm2d(cluster_dim)
        
        self.pool = nn.MaxPool2d(2,2)
        
        ### Decoder
        self.fc_d1 = nn.Linear(cluster_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def encode(self, x):
        x = self.conv_bn1(self.pool(F.relu(self.conv1(x))))
        x = F.relu(self.conv_bn2(self.conv2(x)))
        x = self.pool_bn3(self.pool(F.relu(self.conv_bn3(self.conv3(x)))))
        x = F.relu(self.conv_bn4(self.conv4(x)))
        x = F.relu(self.conv_bn5(self.conv5(x)))
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        
        x = x.view(x.size(0),-1)
        
        return x
        
    def decode(self, z):
        x = self.fc_d1(z)
        x = self.log_softmax(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z