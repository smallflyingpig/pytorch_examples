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
from models import resnet, resnet_pool, resnet_filter, resnet_max_cp, resnet_cn, densenet, densenet_cn, \
    resnet_revert, densenet_revert, resnet_attn_pooling, densenet_attn_pooling
from utils import progress_bar
from trainer import Trainer
import logging
from tensorboardX import SummaryWriter

torch.manual_seed(0)
    
def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--pool_type', type=str, default='pool', help='none, pool, max_cp, filter, or cn')
    parser.add_argument('--model', type=str, default='resnet18', help="model type, resnet18, resnet101, default resnet18")
    parser.add_argument('--log_dir', type=str, default='default')
    parser.add_argument('--data_root', type=str, default="../data/cifar10", help="")
    parser.add_argument('--epoch', type=int, default=350, help="total epoch for the training")
    parser.add_argument('--val_interval', type=int, default=5, help="epoch interval")
    parser.add_argument('--inter_pool', action='store_true', default=False, help="enable inter pool")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="disable the Cuda")
    parser.add_argument('--block_type', type=str, default='bottleneck', help='set the block type')
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--base_dim', type=int, default=64, help='default base dim for resnet, default 64')
    parser.add_argument('--growing_rate', type=int, default=12, help='growing rate')
    
    args = parser.parse_args()
    
    args.device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **param)
    
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **param)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Model
    print('==> Building model..')
    model_dict_all = {
        #'max_cp':{'resnet18':resnet_max_cp.ResNetSimple18, 'resnet110':resnet_max_cp.ResNetSimple110},
        'none':{
            'resnet18':{'model':resnet.ResNetSimple18, 'param': args.block_type}, 
            'resnet110':{'model':resnet.ResNetSimple110, 'param':args.block_type},
            'densenet':{'model':densenet.densenet_cifar, 'param':args.growing_rate}
            },
        'cn':{
            'resnet18':{'model':resnet_cn.ResNetSimple18, 'param':args.block_type}, 
            'resnet110':{'model':resnet_cn.ResNetSimple110, 'param':args.block_type},
            'densenet':{'model':densenet_cn.densenet_cifar, 'param':args.growing_rate}
            },
        'revert':
            {
            'resnet18':{'model':resnet_revert.ResNetSimple18, 'param':args.block_type}, 
            'resnet110':{'model':resnet_revert.ResNetSimple110, 'param':args.block_type},
            'densenet':{'model':densenet_revert.densenet_cifar, 'param':args.growing_rate}
            },
        'attn_pool':
            {
            'resnet18':{'model':resnet_attn_pooling.ResNetSimple18, 'param':args.block_type}, 
            'resnet110':{'model':resnet_attn_pooling.ResNetSimple110, 'param':args.block_type},
            'densenet':{'model':densenet_attn_pooling.densenet_cifar, 'param':args.growing_rate}
                
            }
    }
    net = model_dict_all[args.pool_type][args.model]['model'](model_dict_all[args.pool_type][args.model]['param'])
        
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
    

    def lr_schduler(epoch, lr_gamma=[0.1, 1, 0.1, 0.01]):
        if epoch<2:
            return lr_gamma[0]
        elif epoch<150:
            return lr_gamma[1]
        elif epoch<250:
            return lr_gamma[2]
        else:
            return lr_gamma[3]
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schduler)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1)
    else:
        raise NotImplementedError
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [82, 123], gamma=0.1)
    # Training
    trainer = Trainer(net, trainloader, testloader, optimizer, args.device, criterion, args.logger, args.writer, args.model_dir_full)
    trainer.train(args.epoch, val_interval=args.val_interval, lr_scheduler=scheduler, start_epoch=start_epoch, best_acc=best_acc)
    # load the best model
    # test_dict = trainer.test()
    
    args.logger.info("training and test end, best:{}%".format(trainer.best_acc))


if __name__=="__main__":
    args = parser()
    args = prepare_log(args)
    main(args)