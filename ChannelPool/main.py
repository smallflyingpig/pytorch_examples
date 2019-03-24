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
import logging
from tensorboardX import SummaryWriter

class Trainer(object):
    def __init__(self, model, trainloader, valloader, optimizer, device, criterion, logger, writer, save_model_dir):
        self.model, self.trainloader, self.valloader, self.optimizer, self.device, self.criterion, self.logger, self.writer = \
            model, trainloader, valloader, optimizer, device, criterion, logger, writer
        self.save_model_dir = save_model_dir
        self.best_acc = 0

    def train(self, total_epoch, val_interval, lr_scheduler=None, start_epoch=0):
        for epoch_idx in range(start_epoch, total_epoch):
            if lr_scheduler is not None:
                lr_scheduler.step()
            if epoch_idx % val_interval == 0 or epoch_idx == total_epoch-1:
                acc = self.eval(epoch_idx)
                self.update_best_acc(acc, epoch=epoch_idx)
            self.train_once(epoch_idx)
            
            
    def train_once(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            self.logger.info("epoch:{} | Batch:{} | Type:{} | Loss:{} | Acc:{}% ({:d}/{:d})".format(
                epoch, batch_idx, 'train', train_loss/(batch_idx+1), 100.*correct/total, correct, total
            ))
        train_loss = train_loss/total
        correct = 100. * correct/total
        self.writer.add_scalars('loss', {'train_loss':train_loss}, global_step=epoch)
        self.writer.add_scalars('accu', {'train_accu':correct}, global_step=epoch)
        
        
    def eval(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(self.valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                self.logger.info("epoch:{} |Batch:{} | Type:{} | Loss:{} | Acc:{}% ({:d}/{:d})".format(
                epoch, batch_idx, 'test', test_loss/(batch_idx+1), 100.*correct/total, correct, total
                ))
        test_loss = test_loss/total
        correct = 100. * correct/total
        self.writer.add_scalars('loss', {'test_loss':test_loss}, global_step=epoch)
        self.writer.add_scalars('accu', {'test_accu':correct}, global_step=epoch)
    
        # Save checkpoint.
        acc = 100.*correct/total
        return acc
        
    
    def test(self, loop_num=5, dataloader=None):
        if dataloader is None:
            dataloader = self.valloader
        accu_list = []
        for idx in range(loop_num):
            accu_list.append(self.test(idx))
        accu_np = np.array(accu_list)
        best, mean, std = np.max(accu_np), np.mean(accu_np), np.std(accu_np)
        return {'best':best, 'mean':mean, 'std':std}

    
    def update_best_acc(self, acc, epoch):
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(self.save_model_dir, 'ckpt.t7'))
            self.best_acc = acc
    
    
def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--pool', action='store_true', default=False, help='enable pool')
    parser.add_argument('--model', type=str, default='resnet', help="model type, default resnet")
    parser.add_argument('--log_dir', type=str, default='default')
    parser.add_argument('--data_root', type=str, default="../data/cifar10", help="")
    parser.add_argument('--epoch', type=int, default=350, help="total epoch for the training")
    parser.add_argument('--val_interval', type=int, default=5, help="epoch interval")
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
    handler_stdout.setLevel(logging.ERROR)
    logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[
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
        net = resnet_pool.ResNetSimple18()
    else:
        net = resnet.ResNetSimple18()
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
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.2)
    # Training
    trainer = Trainer(net, trainloader, testloader, optimizer, args.device, criterion, args.logger, args.writer, args.model_dir_full)
    trainer.train(args.epoch, val_interval=args.val_interval, lr_scheduler=scheduler, start_epoch=start_epoch)
    test_dict = trainer.test()
    # for epoch in range(start_epoch, start_epoch+args.epoch):
    #     scheduler.step()
    #     train(epoch)
    #     acc = test(epoch)
    #     update_best_acc(acc, best_acc)
    
    args.logger.info("training and test end, best:{}, mean:{}, std:{}%".format(test_dict['best'] * 100., test_dict['mean']*100., test_dict['std']))


if __name__=="__main__":
    args = parser()
    args = prepare_log(args)
    main(args)