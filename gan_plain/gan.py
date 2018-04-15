# remain no good result

import argparse
import os 
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn

parser = argparse.ArgumentParser(description="plain GAN for mnist")
parser.add_argument("--root", type=str, default="/home/lijiguo/workspace/pytorch_examples", 
                    help="root path for pytorch examples. default /home/lijiguo/workspace/pytorch_examples/")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training data. default 64")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda use")
parser.add_argument("--epoches", type=int, default=10, help="training epoches, default 10")
parser.add_argument("--hidden_dim", type=int, default=1024, help="hidden dimension of gan. (default 1024, 3 layers total)")
parser.add_argument("--z_dim", type=int, default=2, help="dimension of z. default 2")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

#torch.manual_seed(1)
#if args.cuda:
 #   torch.cuda.manual_seed(1)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=True, download=False, 
                    transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=False, download=False,
                    transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)



class Generator(nn.Module):
    """
    generator for GAN
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = nn.Linear(args.z_dim, 64)
        self.dense2 = nn.Linear(64, 128)
        self.dense3 = nn.Linear(128, 256)
        self.dense4 = nn.Linear(256, 28*28)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, z):
        x = self.relu(self.dense1(z))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.dense4(x)
        return self.sigmoid(x).view(-1,1,28,28)


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = nn.Linear(28*28, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)


    def forward(self, z):
        """
        input: z (batch_size x 1 x 28 x 28)
        output: pred (batch_size x 1)
        """
        x = self.dropout1(F.relu(self.dense1(z.view(-1, 28*28))))  #28*28 --> 256
        x = self.dropout2(F.relu(self.dense2(x)))  #256 --> 128
        x = F.sigmoid(self.dense3(x))
        return x

def loss_func(input, target):
    error = F.binary_cross_entropy(input, target)
    return error


model_G = Generator()
model_D = Discriminator()

if args.cuda:
    model_G.cuda()
    model_D.cuda()

optimizer_G = torch.optim.SGD(model_G.parameters(), lr=1e-3, momentum=0.9)
optimizer_D = torch.optim.SGD(model_D.parameters(), lr=1e-4, momentum=0.9)

save_idx = 0
def train(epoch):
    """
    """
    global save_idx

    model_D.train()
    model_G.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        batch_size = data.size(0)
        #real data
        data_real = data
        label_real = torch.Tensor(batch_size).fill_(1)
        data_real, label_real = Variable(data_real), Variable(label_real)

        if args.cuda:
            data_real, label_real = data_real.cuda(), label_real.cuda()

        pred_real = model_D.forward(data_real)
        
        #fake data
        z = Variable(torch.Tensor(np.random.uniform(low=-1, high=1, size=(batch_size, args.z_dim))))
        label_fake = Variable(torch.Tensor(batch_size).fill_(0))
        if args.cuda:
            z = z.cuda()
            label_fake = label_fake.cuda()

        data_fake = model_G.forward(z)

        #update D
        pred_D = model_D.forward(torch.cat([data_real, data_fake]))
        model_D.zero_grad()
        loss_D = loss_func(pred_D[:,0], torch.cat([label_real, label_fake]))
        loss_D.backward(retain_graph=True)
        optimizer_D.step()

        #update G
        pred_G = model_D.forward(data_fake)
        model_G.zero_grad()
        loss_G = loss_func(pred_G[:,0], label_real)
        loss_G.backward()
        optimizer_G.step()

        if batch_idx%100 == 0:
            print("==>epoch:{}, [{}/{}], loss_D:{:.5f}, loss_G:{:.5f}, save idx:{}".format(epoch, batch_idx*len(data), 
                    len(train_loader.dataset), loss_D.cpu().data[0], loss_G.cpu().data[0], save_idx))
            #save image
            if not os.path.exists(os.path.join(args.root, "./gan_plain/sample/")):
                os.system("mkdir {}".format(os.path.join(args.root, "./gan_plain/sample/")))
            
            torchvision.utils.save_image(tensor=data_fake.view(-1,1,28,28).data.cpu(), 
                    filename=os.path.join(args.root, "./gan_plain/sample/", "sample_{}.png".format(save_idx)), nrow=8)
            
            #save model
            if not os.path.exists(os.path.join(args.root, "./gan_plain/model/")):
                os.system("mkdir {}".format(os.path.join(args.root, "./gan_plain/model/")))
            
            torch.save(model_G.state_dict(), f=os.path.join(args.root, "./gan_plain/model/model_G_{}.pytorch".format(save_idx)))
            torch.save(model_D.state_dict(), f=os.path.join(args.root, "./gan_plain/model/model_D_{}.pytorch".format(save_idx)))
            save_idx += 1


if __name__=="__main__":
    for epoch in range(0, args.epoches):
        train(epoch)


        