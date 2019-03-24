'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import numpy as np

# sys.path.append(os.getcwd())
from models import resnet, resnet_pool
from utils import progress_bar
from trainer import Trainer
import logging
from tensorboardX import SummaryWriter

    
def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--pool', action='store_true', default=False, help='enable pool')
    parser.add_argument('--model', type=str, default='resnet18', help="model type, resnet18, resnet101, default resnet18")
    parser.add_argument('--log_dir', type=str, default='default')
    parser.add_argument('--data_root', type=str, default="../data/cifar10", help="")
    parser.add_argument('--epoch', type=int, default=200, help="total epoch for the training")
    parser.add_argument('--val_interval', type=int, default=5, help="epoch interval")
    parser.add_argument('--inter_pool', action='store_true', default=False, help="enable inter pool")
    args = parser.parse_args()
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args

def prepare_log(args):
    args.log_dir_full = os.path.join(os.getcwd(), 'log', args.log_dir)
    args.model_dir_full = os.path.join(os.getcwd(), 'model', args.log_dir)
    if not os.path.exists(args.log_dir_full):
        os.makedirs(args.log_dir_full)
    if not os.path.exists(args.model_dir_full):
        os.makedirs(args.model_dir_full)
    
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG, handlers=[
        logging.FileHandler(os.path.join(args.log_dir_full, 'log.txt')),
    ] )
    logger = logging.getLogger()
    logger.addHandler(handler_stdout)
    logger.info(args)
    
    writer = SummaryWriter(log_dir=args.log_dir_full)
    args.logger, args.writer = logger, writer
    return args


def main(args):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
    param = {"num_workers":4, "pin_memory":True} if args.device=='cuda' else {}
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, **param)
    
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, **param)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Model
    print('==> Building model..')
    
    if args.pool:
        if args.model == 'resnet18':
            net = resnet_pool.ResNetSimple18(args.inter_pool)
        elif args.model == 'resnet110':
            net = resnet_pool.ResNetSimple110(args.inter_pool)
        else:
            raise NotImplementedError
        
    else:
        if args.model == 'resnet18':
            net = resnet.ResNetSimple18()
        elif args.model == 'resnet110':
            net = resnet.ResNetSimple110()
        else:
            raise NotImplementedError
        
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    net = net.to(args.device)
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.model_dir_full), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(args.model_dir_full, 'ckpt.t7'))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [82, 123], gamma=0.1)
    # Training
    trainer = Trainer(net, trainloader, testloader, optimizer, args.device, criterion, args.logger, args.writer, args.model_dir_full)
    trainer.train(args.epoch, val_interval=args.val_interval, lr_scheduler=scheduler, start_epoch=start_epoch, best_acc=best_acc)
    # load the best model
    # test_dict = trainer.test()
    
    args.logger.info("training and test end, best:{}%".format(best_acc))


if __name__=="__main__":
    args = parser()
    args = prepare_log(args)
    main(args)