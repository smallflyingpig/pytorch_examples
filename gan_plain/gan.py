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
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="plain GAN for mnist")
parser.add_argument("--root", type=str, default="/home/lijiguo/workspace/pytorch_examples", 
                    help="root path for pytorch examples. default /home/lijiguo/workspace/pytorch_examples/")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training data. default 64")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda use")
parser.add_argument("--epoches", type=int, default=100, help="training epoches, default 10")
parser.add_argument("--hidden_dim", type=int, default=1024, help="hidden dimension of gan. (default 1024, 3 layers total)")
parser.add_argument("--z_dim", type=int, default=10, help="dimension of z. default 2")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

log_dir = os.path.join(args.root, "./gan_plain/log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

writer = SummaryWriter(log_dir=log_dir)

#torch.manual_seed(1)
#if args.cuda:
 #   torch.cuda.manual_seed(1)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=True, download=True, 
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ]) ),
    batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=False, download=True,
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ]) ),
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
        self.dense4 = nn.Linear(256, 512)
        self.dense5 = nn.Linear(512, 28*28)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    
    def forward(self, z):
        x = self.relu(self.dense1(z))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))
        x = self.relu(self.dense4(x))
        x = self.dense5(x)
        return self.tanh(x).view(-1,1,28,28)


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dense1 = nn.Linear(28*28, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)


    def forward(self, z):
        """
        input: z (batch_size x 1 x 28 x 28)
        output: pred (batch_size x 1)
        """
        x = self.dropout1(F.relu(self.dense1(z.view(-1, 28*28))))  #28*28 --> 512
        x = self.dropout2(F.relu(self.dense2(x)))  #512 --> 256
        x = self.dropout3(F.relu(self.dense3(x)))  #256 --> 128
        x = self.dropout4(F.relu(self.dense4(x)))  #256 --> 128
        x = F.sigmoid(self.dense5(x))
        return x

def loss_func(input, target):
    error = F.binary_cross_entropy(input, target)
    return error


model_G = Generator()
model_D = Discriminator()

if args.cuda:
    model_G.cuda()
    model_D.cuda()

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=2e-4)# torch.optim.SGD(model_G.parameters(), lr=1e-3, momentum=0.9)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=2e-4)# torch.optim.SGD(model_D.parameters(), lr=1e-4, momentum=0.9)

save_idx = 0
batch_cnt = 0
def train(epoch):
    """
    """
    global batch_cnt
    global save_idx

    model_D.train()
    model_G.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        batch_cnt += 1
        batch_size = data.size(0)
        #real data
        data_real = data
        label_real = torch.Tensor(batch_size).fill_(1)
        data_real, label_real = Variable(data_real), Variable(label_real)

        if args.cuda:
            data_real, label_real = data_real.cuda(), label_real.cuda()

        pred_real = model_D.forward(data_real)
        
        #fake data
        z = Variable(torch.randn(size=(batch_size, args.z_dim)))
        label_fake = Variable(torch.Tensor(batch_size).fill_(0))
        if args.cuda:
            z = z.cuda()
            label_fake = label_fake.cuda()

        data_fake = model_G.forward(z)

        #update D
        pred_D = model_D.forward(torch.cat([data_real, data_fake.detach()]))
        model_D.zero_grad()
        loss_D = loss_func(pred_D, torch.cat([label_real, label_fake]))
        loss_D.backward()
        optimizer_D.step()

        #update G
        z = Variable(torch.randn(size=(batch_size, args.z_dim)))
        if args.cuda:
            z = z.cuda()
        data_fake = model_G.forward(z)
        pred_G = model_D.forward(data_fake)
        model_G.zero_grad()
        loss_G = loss_func(pred_G, label_real)
        loss_G.backward()
        optimizer_G.step()

        writer.add_scalars(main_tag="loss", tag_scalar_dict={
            "loss_d":loss_D.cpu().item(),
            "loss_g":loss_G.cpu().item()
        }, global_step=batch_cnt)
        if batch_idx%100 == 0:
            print("==>epoch:{:4d}, [{:5d}/{:5d}], loss_D:{:10.5f}, loss_G:{:10.5f}, save idx:{}".format(epoch, batch_idx*len(data), 
                    len(train_loader.dataset), loss_D.cpu().item(), loss_G.cpu().item(), save_idx))
            #save image
            if not os.path.exists(os.path.join(args.root, "./gan_plain/sample/")):
                os.system("mkdir {}".format(os.path.join(args.root, "./gan_plain/sample/")))
            
            torchvision.utils.save_image(tensor=data_fake.view(-1,1,28,28).data.cpu(), 
                    filename=os.path.join(args.root, "./gan_plain/sample/", "sample_{}.png".format(save_idx)), nrow=8)
            writer.add_image(tag="fake_image", img_tensor=torchvision.utils.make_grid(data_fake, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="real_image", img_tensor=torchvision.utils.make_grid(data_real, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            #save model
            if not os.path.exists(os.path.join(args.root, "./gan_plain/model/")):
                os.system("mkdir {}".format(os.path.join(args.root, "./gan_plain/model/")))
            
            torch.save(model_G.state_dict(), f=os.path.join(args.root, "./gan_plain/model/model_G_{}.pytorch".format(save_idx)))
            torch.save(model_D.state_dict(), f=os.path.join(args.root, "./gan_plain/model/model_D_{}.pytorch".format(save_idx)))
            save_idx += 1


if __name__=="__main__":
    for epoch in range(0, args.epoches):
        train(epoch)


        