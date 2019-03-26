'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.channel_norm import ChannelNorm


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, input_shape):
        super(Bottleneck, self).__init__()
        self.channel_norm1 = ChannelNorm(input_shape[0]*input_shape[1], affine=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.channel_norm2 = ChannelNorm(input_shape[0]*input_shape[1], affine=True)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        

    def forward(self, x):
        out = self.conv1(self.bn1(F.relu(self.channel_norm1(x))))
        out = self.conv2(self.bn2(F.relu(self.channel_norm2(out))))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, input_shape):
        super(Transition, self).__init__()
        self.channel_norm = ChannelNorm(input_shape[0]*input_shape[1], affine=True)
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.conv(self.bn(F.relu(self.channel_norm(x))))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, input_shape=(32,32)):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False) # (64,32,32)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], input_shape) #(64+12*6)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, input_shape)
        num_planes = out_planes
        input_shape = (input_shape[0]//2, input_shape[1]//2)

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], input_shape)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, input_shape)
        num_planes = out_planes
        input_shape = (input_shape[0]//2, input_shape[1]//2)

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], input_shape)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, input_shape)
        num_planes = out_planes
        input_shape = (input_shape[0]//2, input_shape[1]//2)

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], input_shape)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.channel_norm = ChannelNorm(input_shape[0]*input_shape[1], affine=True)
        self.linear = nn.Linear(num_planes, num_classes)
        

    def _make_dense_layers(self, block, in_planes, nblock, input_shape):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, input_shape=input_shape))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(self.bn(F.relu(self.channel_norm(out))), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar(growth_rate=12):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, input_shape=(32,32))

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()