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

# sys.path.append(os.getcwd())
from models import resnet, resnet_pool
from utils import progress_bar
import logging
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--pool', action='store_true', default=False, help='enable pool')
parser.add_argument('--model', type=str, default='resnet', help="model type, default resnet")
parser.add_argument('--log_dir', type=str, default='default')
parser.add_argument('--data_root', type=str, default="../data/cifar10", help="")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


log_dir_full = os.path.join(os.getcwd(), 'log', args.log_dir)
model_dir_full = os.path.join(os.getcwd(), 'model', args.log_dir)
if not os.path.exists(log_dir_full):
    os.makedirs(log_dir_full)
if not os.path.exists(model_dir_full):
    os.makedirs(model_dir_full)

handler_stdout = logging.StreamHandler(sys.stdout)
handler_stdout.setLevel(logging.ERROR)
logging.basicConfig(format='%(message)s', level=logging.INFO, handlers=[
    logging.FileHandler(os.path.join(log_dir_full, 'log.txt')),
] )
logger = logging.getLogger()
logger.addHandler(handler_stdout)
logger.info(args)

writer = SummaryWriter(log_dir=log_dir_full)

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')

if args.pool:
    net = resnet_pool.ResNet18()
else:
    net = resnet.ResNet18()
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
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(model_dir_full), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(model_dir_full, 'ckpt.t7'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        logger.info("epoch:{} | Batch:{} | Type:{} | Loss:{} | Acc:{}% ({:d}/{:d})".format(
            epoch, batch_idx, 'train', train_loss/(batch_idx+1), 100.*correct/total, correct, total
        ))
    train_loss = train_loss/total
    correct = 100. * correct/total
    writer.add_scalars('loss', {'train_loss':train_loss}, global_step=epoch)
    writer.add_scalars('accu', {'train_accu':correct}, global_step=epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            logger.info("epoch:{} |Batch:{} | Type:{} | Loss:{} | Acc:{}% ({:d}/{:d})".format(
            epoch, batch_idx, 'test', test_loss/(batch_idx+1), 100.*correct/total, correct, total
        ))
    test_loss = test_loss/total
    correct = 100. * correct/total
    writer.add_scalars('loss', {'test_loss':test_loss}, global_step=epoch)
    writer.add_scalars('accu', {'test_accu':correct}, global_step=epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_dir_full, 'ckpt.t7'))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
logger.info("training end, best accu: {}%".format(best_acc * 100.))