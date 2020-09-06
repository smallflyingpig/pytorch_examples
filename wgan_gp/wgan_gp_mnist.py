# remain no good result

import argparse
import os 
import numpy as np
import tqdm 
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn
from torch import autograd

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="plain GAN for mnist")
parser.add_argument("--root", type=str, default="/home/lijiguo/workspace/pytorch_examples", 
                    help="root path for pytorch examples. default /home/lijiguo/workspace/pytorch_examples/")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training data. default 64")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda use")
parser.add_argument("--epoches", type=int, default=50, help="training epoches, default 10")
parser.add_argument("--hidden_dim", type=int, default=1024, help="hidden dimension of gan. (default 1024, 3 layers total)")
parser.add_argument("--z_dim", type=int, default=10, help="dimension of z. default 2")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("cuda in available, use cuda")
else:
    print("cuda in not available, use cpu")

model_dir = "./wgan_gp"
log_dir = os.path.join(args.root, model_dir, "./log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

writer = SummaryWriter(log_dir=log_dir)

#torch.manual_seed(1)
#if args.cuda:
 #   torch.cuda.manual_seed(1)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=True, download=False, 
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ]) ),
    batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=False, download=False,
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
    def __init__(self, noise_dim=10, channel=1):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.channel = channel
        self.net = nn.Sequential(
            # batch x 10 x 1 x 1 --> batch x 512 x 3 x 3
            nn.ConvTranspose2d(self.noise_dim, 512, 3, 1, 0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # batch x 512 x 3 x 3 --> batch x 256 x 6 x 6
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # batch x 256 x 6 x 6 --> batch x 128 x 12 x 12
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # batch x 128 x 12 x 12 --> batch x  64 x 24 x 24
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # batch x 64 x 24 x 24 --> batch x channel x 28 x 28
            nn.ConvTranspose2d(64, self.channel, 5, 1, 0)
        )
        self.output = nn.Tanh()
    
    def forward(self, z):
        x = self.net(z)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self, channel=1):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.net = nn.Sequential(
            # 3 x 28 x 28 --> 64 x 24 x 24
            nn.Conv2d(self.channel, 64, 5, 1, 0),
            nn.ReLU(True),
            # 64 x 24 x 24 --> 128 x 12 x 12
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            # 128 x 12 x 12 --> 256 x 6 x 6
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(True),
            # 256 x 6 x 6 --> 512 x 3 x 3
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 0),
            # no sigmoid here
            #nn.Sigmoid()
        )


    def forward(self, z):
        """
        input: z (batch_size x 1 x 28 x 28)
        output: pred (batch_size x 1)
        """
        x_feature = self.net(z)
        x = self.output(x_feature)
        return x.squeeze()

def loss_func(input, target):
    error = F.binary_cross_entropy(input, target)
    return error


def grad_penalty(model_D, real_data, fake_data):
    # real_data: batch * channel * W * H
    batch_size = real_data.shape[0]
    alpha = torch.rand((batch_size,1))
    alpha = alpha.unsqueeze(2).unsqueeze(3).expand(real_data.size())
    #alpha = alpha.expand_as(real_data)
    if args.cuda:
        alpha = alpha.cuda() 
    interpolations = alpha * real_data + (1-alpha)*fake_data
    if args.cuda:
        interpolations = interpolations.cuda()

    interpolations.requires_grad_(True)
    dis_interpolations = model_D.forward(interpolations)
    grad_outputs = torch.ones_like(dis_interpolations)
    if args.cuda:
        grad_outputs = grad_outputs.cuda()
    grad = autograd.grad(outputs=dis_interpolations, inputs=interpolations, 
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_penalty = ((grad-1)**2).mean() * 10
    return grad_penalty


model_G = Generator(noise_dim=args.z_dim)
model_D = Discriminator()

if args.cuda:
    model_G.cuda()
    model_D.cuda()

# use Adam optimizer
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=1e-4, betas=(0.5, 0.9))# torch.optim.SGD(model_G.parameters(), lr=1e-3, momentum=0.9)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-4, betas=(0.5, 0.9))# torch.optim.SGD(model_D.parameters(), lr=1e-4, momentum=0.9)

D_iter = 5
D_iter_start = 20

save_idx = 0
batch_cnt = 0
def train(epoch):
    """
    """
    global batch_cnt
    global save_idx

    model_D.train()
    model_G.train()
    loader_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        batch_cnt += 1
        batch_size = data.size(0)
        #real data
        data_real = data
        data_real = Variable(data_real)

        if args.cuda:
            data_real = data_real.cuda()
        
        train_D_flag = False
        train_G_flag = False

        if batch_cnt % 100 < D_iter_start:
            train_D_flag = True
            train_G_flag = False
        elif batch_cnt % 2 == 0:
            train_D_flag = True
            train_G_flag = True
        else:
            train_D_flag = True
            train_G_flag = False
        
        if train_D_flag:
            #fake data
            z = torch.randn(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_(True)
            if args.cuda:
                z = z.cuda()
    
            data_fake = model_G.forward(z)
    
            #update D
            model_D.zero_grad()
            pred_D_real = model_D.forward(data_real)
            pred_D_fake = model_D.forward(data_fake.detach())
            loss_D_real = pred_D_real.mean() #loss_func(pred_D_real, label_real)
            loss_D_fake = pred_D_fake.mean() #loss_func(pred_D_fake, label_fake)
            # no log here
            loss_D_gp = grad_penalty(model_D, data_real, data_fake.detach())
            loss_D = -loss_D_real + loss_D_fake + loss_D_gp
    
            loss_D.backward()
            optimizer_D.step()
            # no param clip for WGAN-GP
            #for p in model_D.parameters():
            #    p.data.clamp_(-0.02, 0.02)
        if train_G_flag:
            #update G
            z = torch.randn(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_(True)
            if args.cuda:
                z = z.cuda()
            data_fake = model_G.forward(z)
            pred_G = model_D.forward(data_fake)
            model_G.zero_grad()
            # no log here
            loss_G = -pred_G.mean() 
            loss_G.backward()
            optimizer_G.step()
    
            writer.add_scalars(main_tag="loss", tag_scalar_dict={
                "loss_d":loss_D.cpu().item(),
                "loss_g":loss_G.cpu().item(),
                "loss_d_gp":loss_D_gp.cpu().item()
            }, global_step=batch_cnt)

        if train_G_flag and train_D_flag and batch_cnt%50 == 0:
            print("==>epoch:{:4d}, [{:5d}/{:5d}], loss_D:{:10.5f}, loss_G:{:10.5f}, save idx:{}".format(epoch, batch_idx*len(data), 
                    len(train_loader.dataset), loss_D.cpu().item(), loss_G.cpu().item(), save_idx))
            #save image
            if not os.path.exists(os.path.join(args.root, model_dir, "./sample/")):
                os.system("mkdir {}".format(os.path.join(args.root, model_dir, "./sample/")))
            
            torchvision.utils.save_image(tensor=data_fake.view(-1,1,28,28).data.cpu(), 
                    filename=os.path.join(args.root, model_dir, "./sample/", "sample_{}.png".format(save_idx)), nrow=8)
            writer.add_image(tag="fake_image", img_tensor=torchvision.utils.make_grid(data_fake, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="real_image", img_tensor=torchvision.utils.make_grid(data_real, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            #save model
            if not os.path.exists(os.path.join(args.root, model_dir, "./model/")):
                os.system("mkdir {}".format(os.path.join(args.root, model_dir, "./model/")))
            
            torch.save(model_G.state_dict(), f=os.path.join(args.root, model_dir, "./model/model_G_{}.pytorch".format(save_idx)))
            torch.save(model_D.state_dict(), f=os.path.join(args.root, model_dir, "./model/model_D_{}.pytorch".format(save_idx)))
            save_idx += 1


if __name__=="__main__":
    for epoch in range(0, args.epoches):
        train(epoch)


        