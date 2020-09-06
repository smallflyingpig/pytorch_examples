import os
import argparse
import torch
import tqdm 
from tensorboardX import SummaryWriter
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser(description="MNIST pytorch")
parser.add_argument("--batch_size", type=int, default=64, metavar='N', help="batch size (default 64)")
parser.add_argument("--learning_rate", type=float, default=1e-2, metavar="LR", help="learning rate for training, default 10^-2")
parser.add_argument("--no_cuda",action="store_true",default=False, help="disable the cuda")
parser.add_argument("--epoches", type=int, default=10, help="set the training epoches, default 10")
parser.add_argument("--root", type=str, default="/home/lijiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

model_dir = "./mnist_classification"
log_dir = os.path.join(args.root, model_dir, "./log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

writer = SummaryWriter(log_dir=log_dir)

train_loader=torch.utils.data.DataLoader( 
            torchvision.datasets.MNIST(root=args.root+"./data/mnist",train=True, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
            batch_size=args.batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader( 
            torchvision.datasets.MNIST(root=args.root+"./data/mnist",train=False, 
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])),
            batch_size=args.batch_size,shuffle=True)

class Net(torch.nn.Module):
    """
    module defination
    """
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)  #1x28x28-->10x24x24
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)  #10x12x12-->20x8x8
        self.dropout = nn.Dropout2d()

        self.linear1 = nn.Linear(320, 50)
        self.linear2 = nn.Linear(50, 10)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        """
        param: input_data, 1x28x28 variable
        return: ouput_data, 1x10 variable
        """
        assert(input_data.shape[-3:] == torch.Size([1,28,28]))  #batch_sizex1x28x28
        self.data = F.relu(F.max_pool2d(self.conv1(input_data),2)) #batch_sizex10x12x12
        self.data = F.relu(F.max_pool2d(self.dropout(self.conv2(self.data)),2)) #batch_sizex20x4x4
        self.data = F.relu(self.linear1(self.data.view(-1, 320)))  #batch_sizex50
        self.data = self.linear2((self.data))  #batch_sizex10
        self.data = self.log_softmax(self.data)
        return self.data

    def get_feature(self, input_data):
        assert(input_data.shape[-3:] == torch.Size([1,28,28]))  #batch_sizex1x28x28
        data = F.relu(F.max_pool2d(self.conv1(input_data),2)) #batch_sizex10x12x12
        data = self.dropout(self.conv2(data)) #batch_sizex10x8x8
        return data


model = Net()
optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
loss_func = torch.nn.NLLLoss()
save_idx = 0

def train(epoch):
    """
    param: 
    return:
    """
    model.train()
    loader_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        data, label = data.requires_grad_(), label.requires_grad_(False)

        torch.unsqueeze(data,2)
        pred = model.forward(data)
        loss = loss_func(pred, label)

        model.zero_grad()
        loss.backward()

        optimizer.step()

        loader_bar.set_description("[{:5s}] Epoch:{:5d}, loss:{:10.5f}".format("train", epoch, loss.cpu().item()))
        if batch_idx%100==0:
            #torch.save(model.state_dict(),"./model/model_{}.pytorch".format(save_idx))
            #save_idx += 1
            pass


def test(epoch):
    global save_idx
    model.eval()
    test_loss = 0
    accuracy = 0
    loader_bar = tqdm.tqdm(test_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        data, label = data.requires_grad_(False), label.requires_grad_(False)
        torch.unsqueeze(data,2)
        pred = model.forward(data)
        loss = loss_func(pred, label)
        test_loss += loss.cpu().item()
        pred_label = pred.data.max(1,keepdim=True)[1]
        accuracy += pred_label.eq(label.data.view_as(pred_label)).long().cpu().sum()
        loader_bar.set_description("[{:5s}] Epoch:{:5d}, loss:{:10.5f}".format("test", epoch, loss.cpu().item()))

    accuracy = float(accuracy)/len(test_loader.dataset)

    print("[{:5s}] Epoch:{:5d}, loss:{:10.5f}, accuracy:{:10.5f}".format("test", epoch, test_loss, accuracy))

    if not os.path.exists(os.path.join(args.root, model_dir, "./model/")):
        print("path ({}) does not exist, create it".format(args.root+"./mnist_classification/model/"))
        os.system("mkdir {}".format(args.root+"./mnist_classification/model"))

    torch.save(model.state_dict(),args.root+"./mnist_classification/model/"+"./model_{}.pytorch".format(save_idx))
    save_idx += 1
    

if __name__ == "__main__":
    for epoch in range(0, args.epoches):
        train(epoch)
        test(epoch)




