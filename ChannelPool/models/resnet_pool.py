'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.channel_pool import GroupChannelPooling

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, pool=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pool_layer = nn.Sequential()
        self.pool_ratio = 1
        if pool:
            self.pool_layer = nn.Sequential(
                GroupChannelPooling(2,2)
            )
            self.pool_ratio = 2

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes//self.pool_ratio:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool_layer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, pool=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.pool_layer = nn.Sequential()
        self.pool_ratio = 1
        if pool:
            self.pool_layer = nn.Sequential(
                GroupChannelPooling(2,2)
            )
            self.pool_ratio = 2

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes//self.pool_ratio:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes//self.pool_ratio, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes//self.pool_ratio)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.pool_layer(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_dim=64):
        super(ResNet, self).__init__()
        self.in_planes = base_dim

        self.conv1 = nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_dim)
        self.layer1 = self._make_layer(block, base_dim, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_dim*2, num_blocks[1], stride=2, channel_pool_flag=True)
        self.layer3 = self._make_layer(block, base_dim*2, num_blocks[2], stride=2, channel_pool_flag=True)
        self.layer4 = self._make_layer(block, base_dim*2, num_blocks[3], stride=2, channel_pool_flag=True)
        self.linear = nn.Linear(base_dim*block.expansion//2, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, channel_pool_flag=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        if channel_pool_flag:
            layers.append(GroupChannelPooling(kernel_size=2, stride=2))
            self.in_planes = self.in_planes//2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2], base_dim=64)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])



class ResNetSimple(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_dim=16, inter_pool=False):
        super(ResNetSimple, self).__init__()
        self.in_planes = base_dim

        self.conv1 = nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_dim)
        self.layer1 = self._make_layer(block, base_dim, num_blocks[0], stride=1, channel_pool_flag=True, inter_pool=inter_pool)
        self.layer2 = self._make_layer(block, base_dim*2, num_blocks[1], stride=2, channel_pool_flag=True, inter_pool=inter_pool)
        self.layer3 = self._make_layer(block, base_dim*4, num_blocks[2], stride=2, channel_pool_flag=True, inter_pool=inter_pool)
        self.linear = nn.Linear(base_dim*4*block.expansion//2, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, channel_pool_flag=False, inter_pool=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, pool=inter_pool))
            self.in_planes = planes * block.expansion
            if channel_pool_flag:
                if not inter_pool:
                    layers.append(GroupChannelPooling(kernel_size=2, stride=2))
                self.in_planes = self.in_planes//2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetSimple18(inter_pool=False):
    return ResNetSimple(Bottleneck, [3,3,3], inter_pool=inter_pool)

def ResNetSimple110(inter_pool=False):
    return ResNetSimple(Bottleneck, [18,18,18], inter_pool=inter_pool)



def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()