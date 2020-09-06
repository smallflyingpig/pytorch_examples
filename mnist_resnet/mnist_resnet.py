import os
import sys
import argparse
import logging
import torch
import tqdm 
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models.resnet as resnet
import numpy as np
from easydict import EasyDict

sys.path.append(os.getcwd())
from utils.utils import BestRecoder, Timer

parser = argparse.ArgumentParser(description="MNIST Resnet pytorch")
parser.add_argument("--batch_size", type=int, default=64, metavar='N', help="batch size (default 64)")
parser.add_argument("--learning_rate", type=float, default=1e-2, metavar="LR", help="learning rate for training, default 10^-2")
parser.add_argument("--no_cuda",action="store_true",default=False, help="disable the cuda")
parser.add_argument("--epoches", type=int, default=30, help="set the training epoches, default 10")
parser.add_argument("--root", type=str, default="/home/lijiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")
parser.add_argument("--order", type=int, default=1, help="order for Tylor net")        
parser.add_argument("--lr_decay", type=float, default=0.1, help="decay each epoch")        
parser.add_argument("--weight_decay", type=float, default=4e-5, help="weight decay for optimizer")        
parser.add_argument("--block_type", type=str, default="res", help="block type") 
parser.add_argument("--log_file", type=str, default="mnist_resnet.log", help="log file") 
parser.add_argument("--dataset", type=str, default="mnist", help="dataset, mnist or cifar10") 


# args
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

args.root = os.getcwd()

runtime_cfg = EasyDict()

runtime_cfg.model_dir = os.path.split(os.path.dirname(__file__))[-1]  # "./mnist_resnet"
runtime_cfg.log_dir = os.path.join(args.root, runtime_cfg.model_dir, "./log")
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[
    logging.FileHandler(os.path.join(runtime_cfg.log_dir, args.log_file)),
    logging.StreamHandler(sys.stdout)
] )
runtime_cfg.logger = logging.getLogger()

if not os.path.exists(runtime_cfg.log_dir):
    os.mkdir(runtime_cfg.log_dir)

runtime_cfg.writer = SummaryWriter(log_dir=runtime_cfg.log_dir)



# 3*3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class PolyBlock(nn.Module):
    """
    x_{n+1} = x_n + f(x_{n+1}), one conv for f
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PolyBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)
        
        out1 = self.relu(residual + out)
        
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out = self.relu(out1 + residual)
        return out

class PolyBlockV1(nn.Module):  # implicit Euler
    """
    x_{n+1} = x_n + f(x_{n+1}), two conv for f
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PolyBlockV1, self).__init__()
        self.res_conv1 = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv2 = nn.Sequential(
            conv3x3(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = self.res_conv1(x)
        # out = self.relu(out)
        
        out1 = self.relu(residual + out)
        
        out1 = self.res_conv2(out1)
        out = self.relu(out1 + residual)
        return out

class PolyBlockV2(nn.Module):  # implicit Euler
    """
    x_{n+1} = x_n + (f(x_n)+f(x_{n+1}))/2
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PolyBlockV2, self).__init__()
        self.res_conv1 = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv2 = nn.Sequential(
            conv3x3(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = self.res_conv1(x)
        # out = self.relu(out)
        
        out1 = self.relu(residual + out)
        
        out1 = self.res_conv2(out1)
        out_final = self.relu((out1+out)/2 + residual)
        return out_final


class PolyBlockV3(nn.Module):  # implicit Euler
    """
    x_{n+1} = x_n + (f(x_n)+f(x_{n+1}))/2
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PolyBlockV3, self).__init__()
        self.res_conv1 = nn.Sequential(
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv2 = nn.Sequential(
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = self.res_conv1(residual)
        # out = self.relu(out)
        
        out1 = self.relu(residual + out)
        
        out1 = self.res_conv2(out1)
        out_final = self.relu((out1+out)/2 + residual)
        return out_final

class RKBlock(nn.Module):  
    """
    Runge-Kutta
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(RKBlock, self).__init__()
        self.res_conv = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv_h_2 = nn.Sequential(
            conv3x3(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.res_conv_h = nn.Sequential(
            conv3x3(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        
        K1 = self.res_conv(x)
        K2 = self.res_conv_h_2(self.relu(residual+K1/2))
        K3 = self.res_conv_h_2(self.relu(residual+K2/2))
        K4 = self.res_conv_h(self.relu(residual+K3))

        out_final = self.relu(residual+K1/6 + K2/3 + K3/3 + K4/6)
        
        return out_final


class TylorBlockV1(nn.Module):  # Tylor
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(TylorBlockV1, self).__init__()
        self.res_conv1 = nn.Sequential(
            conv3x3(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.res_convs = []
        self.order = args.order
        for idx in range(self.order-1):
            block = nn.Sequential(
                    conv3x3(out_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    conv3x3(out_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    )
            if args.cuda:
                block = block.cuda()
            self.res_convs.append(
                block
            )
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = self.res_conv1(x)
        # out = self.relu(out)
        
        # tylors = [residual, out]
        base = 1
        out_final = residual + out
        for idx in range(self.order-1):
            base *= (idx+1)
            out = self.res_convs[idx](self.relu(out))
            out_final += (out/base)
        out_final = self.relu(out_final)
        return out_final

class DenseBlock(nn.Module):  # Dense
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(DenseBlock, self).__init__()
        self.res_convs = []
        self.order = args.order
        for idx in range(self.order):
            block = nn.Sequential(
                    conv3x3(out_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    conv3x3(out_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    )
            if args.cuda:
                block = block.cuda()
            self.res_convs.append(
                block
            )
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)

        out = residual
        # out = self.relu(out)
        
        # tylors = [residual, out]
        out_final = residual
        for idx in range(self.order):
            out = self.res_convs[idx](self.relu(out))
            out_final += (out)
        out_final = self.relu(out_final)
        return out_final        



# ResNet
class ResNet_Mnist(nn.Module):
    def __init__(self, block, layers, num_classes=10, ndim=16):
        super(ResNet_Mnist, self).__init__()
        self.in_channels = ndim
        self.conv = conv3x3(args.input_channel, ndim)
        self.bn = nn.BatchNorm2d(ndim)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, ndim, layers[0])
        self.layer2 = self.make_layer(block, ndim*2, layers[1], 2)
        self.layer3 = self.make_layer(block, ndim*4, layers[2], 2)
        self.layer4 = self.make_layer(block, ndim*8, layers[3], 2)
        self.avg_pool = nn.AvgPool2d(3)
        self.fc = nn.Linear(ndim*8, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out






if args.dataset == "mnist":
    train_dataset = torchvision.datasets.MNIST(root=os.path.join(args.root, "./data/mnist"),train=True, download=True, 
                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
                        ]))
    test_dataset = torchvision.datasets.MNIST(root=os.path.join(args.root, "./data/mnist"),train=False, 
                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
                        ]))
    args.input_channel = 1
    args.num_classes = 10
    ResNet = ResNet_Mnist
elif args.dataset == "cifar10":
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(args.root, "./data/cifar10"),train=True, download=True, 
                transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                        ]))
    test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(args.root, "./data/cifar10"),train=False, 
                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
                        ]))
    args.input_channel = 3
    args.num_classes = 10
    ResNet = ResNet_Mnist
else:
    raise NotImplementedError

param = {"num_workers": 8, "pin_memory": True} if args.cuda else {"num_workers": 8}
train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **param)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **param)



def l1_loss(params):
    param_sum = torch.zeros(1)
    param_cnt = 0
    for param in params:
        param_sum += torch.pow(param,2).mean()*0.5
        param_cnt += 1
    return param_sum/param_cnt

def my_loss(pred, target):
    loss = -pred + target*torch.log(pred) - (1-target)*torch.log(1-pred)
    return loss

blocks = {
    "res": ResidualBlock,
    "poly": PolyBlock,
    "poly1": PolyBlockV1,
    "RK": RKBlock, 
    "poly2": PolyBlockV2,
    "poly3": PolyBlockV3,
    "tylor": TylorBlockV1
}

# Create ResNet
net_args = {
    "block": blocks[args.block_type], #PolyBlockV1, # ResidualBlock,
    "layers": [2, 2, 2, 2],
    "num_classes": args.num_classes,
    "ndim": 16 if args.dataset=='mnist' else 64
}
model = ResNet(**net_args)
if args.cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()

runtime_cfg.lr_decay_epoches = [int(args.epoches//4), int(args.epoches//2), int(args.epoches*3//4)]
runtime_cfg.current_lr = args.learning_rate

def train(epoch, runtime_cfg):
    """
    param: 
    return:
    """
    model.train()
    train_loss = 0
    loader_bar = tqdm.tqdm(train_loader)
    # lr decay
    if epoch in runtime_cfg.lr_decay_epoches:
        runtime_cfg.current_lr *= args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = runtime_cfg.current_lr
        runtime_cfg.logger.info("decay lr to {}".format(runtime_cfg.current_lr))
    for batch_idx, (data, label) in enumerate(loader_bar):
        data, label = data.requires_grad_(), label.requires_grad_(False)
        if args.cuda:
            data, label = data.cuda(), label.cuda()
            
        torch.unsqueeze(data, 2)
        pred = model.forward(data)
        loss = loss_func(pred, label) #+ 100 * l1_loss(model.linear1.parameters())

        model.zero_grad()
        loss.backward()
        optimizer.step()
        loader_bar.set_description("[{:5s}] Epoch:{:5d}, loss:{:10.5f}".format("train", epoch, loss.cpu().item()))
        train_loss += loss.cpu().item()
    loader_bar.close()
    train_loss = train_loss/len(train_loader)
    runtime_cfg.logger.info("[{:5s}] Epoch:{:5d}, loss:{:10.5f}".format("train", epoch, train_loss))


def test(epoch, runtime_cfg):
    model.eval()
    test_loss = 0
    accuracy = 0
    loader_bar = tqdm.tqdm(test_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        data, label = data.requires_grad_(False), label.requires_grad_(False)
        if args.cuda:
            data, label = data.cuda(), label.cuda()

        torch.unsqueeze(data, 2)
        pred = model.forward(data)
        loss = loss_func(pred, label) #+ l1_loss(model.linear1.parameters())
        test_loss += loss.cpu().item()
        pred_label = pred.data.max(1,keepdim=True)[1]
        accuracy += pred_label.eq(label.data.view_as(pred_label)).long().cpu().sum()
        loader_bar.set_description("[{:5s}] Epoch:{:5d}, loss:{:10.5f}".format("test", epoch, loss.cpu().item()))

    accuracy = float(accuracy)/len(test_loader.dataset)
    test_loss = test_loss/len(test_loader)

    loader_bar.close()
    runtime_cfg.logger.info("[{:5s}] Epoch:{:5d}, loss:{:10.5f}, accuracy:{:10.5f}".format("test", epoch, test_loss, accuracy))
    
    model_path = os.path.join(args.root, runtime_cfg.model_dir, "./model")
    if not os.path.exists(model_path):
        logging.warning("path ({}) does not exist, create it".format(model_path))
        os.system("mkdir {}".format(model_path))

    torch.save(model.state_dict(),os.path.join(model_path, "./model_{}.pytorch".format(epoch)))
    return accuracy
    


if __name__ == "__main__":
    runtime_cfg.logger.info(args)
    best_recoder = BestRecoder(init=0)
    for epoch in range(0, args.epoches):
        train(epoch, runtime_cfg)
        accu = test(epoch, runtime_cfg)
        best_recoder.update(accu)

    runtime_cfg.logger.info("best accu: {:10.5f}".format(best_recoder.best))




