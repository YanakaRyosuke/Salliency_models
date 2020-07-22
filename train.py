#!/usr/local/python3/bin/python3
# -*- coding: utf-8 -*-
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

import pdb
import cv2
from PIL import Image

class NeuralNet (nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 32x32size & 3チャネル, 隠れ層のunit数は、600
        self.conv1 = nn.Conv2d(3,  32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2) 
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 == nn.Linear(128*10*10, 48*48*2)
        self.pool_size = 2
        self.fc2 = nn.Linear(2304,2304)
            
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1,1228*10*10)
        x = self.fc(x)
        assert x.shape[-1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        
        x = self.fc2(x)
        return x
