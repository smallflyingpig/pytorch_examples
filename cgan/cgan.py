# remain no good result

import argparse
import os 
import tqdm 
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.nn as nn
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="plain GAN for mnist")
parser.add_argument("--root", type=str, default="/home/ubuntu/lijiguo/workspace/pytorch_examples", 
                    help="root path for pytorch examples. default /home/lijiguo/workspace/pytorch_examples/")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training data. default 64")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda use")
parser.add_argument("--epoches", type=int, default=100, help="training epoches, default 10")
parser.add_argument("--hidden_dim", type=int, default=1024, help="hidden dimension of gan. (default 1024, 3 layers total)")
parser.add_argument("--z_dim", type=int, default=10, help="dimension of z. default 10")
parser.add_argument("--condition_dim", type=int, default=10, help="dimension of c. default 10")
parser.add_argument("--condition_embedding_dim", type=int, default=10, help="embedding dimension of c. default 2")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

model_dir = "./cgan"
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
    def __init__(self, noise_dim=10, condition_dim=10, condition_embedding_dim=2, channel=1):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim=condition_dim
        self.condition_embedding_dim = condition_embedding_dim
        self.channel = channel
        self.net = nn.Sequential(
            # batch x 10 x 1 x 1 --> batch x 512 x 3 x 3
            nn.ConvTranspose2d(self.noise_dim+self.condition_embedding_dim, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # batch x 512 x 3 x 3 --> batch x 256 x 6 x 6
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # batch x 256 x 6 x 6 --> batch x 128 x 12 x 12
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # batch x 128 x 12 x 12 --> batch x  64 x 24 x 24
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # batch x 64 x 24 x 24 --> batch x channel x 28 x 28
            nn.ConvTranspose2d(64, self.channel, 5, 1, 0)
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_embedding_dim, bias=False),
            nn.BatchNorm1d(self.condition_embedding_dim),
            nn.ReLU(True)
        )
        self.output = nn.Tanh()
    
    def forward(self, z, c):
        c = self.embedding_layer(c)
        z_c = torch.cat([z,c.unsqueeze(2).unsqueeze(3)], 1)
        x = self.net(z_c)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self, condition_dim, condition_embedding_dim, image_size=28, channel=1):
        super(Discriminator, self).__init__()
        self.condition_dim = condition_dim
        self.condition_embedding_dim = condition_embedding_dim
        self.image_size = image_size
        self.channel = channel
        self.net = nn.Sequential(
            # 3 x 28 x 28 --> 64 x 24 x 24
            nn.Conv2d(self.channel, 64, 5, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # 64 x 24 x 24 --> 128 x 12 x 12
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 128 x 12 x 12 --> 256 x 6 x 6
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 256 x 6 x 6 --> 512 x 3 x 3
            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.image_size_temp = (self.image_size-4)//8
        self.project_layer = nn.Sequential(
            nn.Linear(512*self.image_size_temp**2, 512*3**2, bias=False),
            nn.BatchNorm1d(512*3**2),
            nn.LeakyReLU(0.2, True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(512+self.condition_embedding_dim, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 3, 1, 0),
            nn.Sigmoid()
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.condition_dim, self.condition_embedding_dim, False),
            nn.BatchNorm1d(self.condition_embedding_dim),
            nn.LeakyReLU(0.2, True)
        )


    def forward(self, z, c):
        """
        input: z (batch_size x 1 x 28 x 28)
        output: pred (batch_size x 1)
        """
        x_feature_temp = self.net(z).view(-1, 512*self.image_size_temp**2)
        x_feature = self.project_layer(x_feature_temp).view(-1, 512, 3, 3)
        c_embedding = self.embedding_layer(c).unsqueeze(2).unsqueeze(3).repeat([1,1,3,3])
        x_feature_c = torch.cat([x_feature, c_embedding], 1)
        x = self.output(x_feature_c)
        return x.squeeze()

def loss_func(input, target):
    error = F.binary_cross_entropy(input, target)
    return error


model_G = Generator(noise_dim=args.z_dim, condition_dim=args.condition_dim, condition_embedding_dim=args.condition_embedding_dim)
model_D = Discriminator(condition_dim=args.condition_dim, condition_embedding_dim=args.condition_embedding_dim)

if args.cuda:
    model_G.cuda()
    model_D.cuda()

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=2e-4, betas=(0.5, 0.999))# torch.optim.SGD(model_G.parameters(), lr=1e-3, momentum=0.9)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))# torch.optim.SGD(model_D.parameters(), lr=1e-4, momentum=0.9)

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
        label_real = torch.Tensor(batch_size).requires_grad_(False).fill_(1)
        data_real = Variable(data_real)
        condition = torch.eye(10)[label].reshape(batch_size, 10)

        if args.cuda:
            data_real, label_real, condition = data_real.cuda(), label_real.cuda(), condition.cuda()

        pred_real = model_D.forward(data_real, condition)
        
        #fake data
        z = torch.randn(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_()
        label_fake = torch.Tensor(batch_size).requires_grad_(False).fill_(0)
        if args.cuda:
            z = z.cuda()
            label_fake = label_fake.cuda()

        data_fake = model_G.forward(z, condition)

        #update D
        model_D.zero_grad()
        pred_D_real = model_D.forward(data_real, condition)
        pred_D_fake = model_D.forward(data_fake.detach(), condition)
        loss_D_real = loss_func(pred_D_real, label_real)
        loss_D_fake = loss_func(pred_D_fake, label_fake)
        loss_D = (loss_D_real+loss_D_fake)/2

        #pred_D = model_D.forward(torch.cat([data_real, data_fake]))
        #loss_D = loss_func(pred_D, torch.cat([label_real, label_fake]))

        loss_D.backward()
        optimizer_D.step()

        #update G
        z = torch.randn(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_()
        if args.cuda:
            z = z.cuda()
        data_fake = model_G.forward(z, condition)
        pred_G = model_D.forward(data_fake, condition)
        model_G.zero_grad()
        loss_G = loss_func(pred_G, label_real)
        loss_G.backward()
        optimizer_G.step()
        
        loader_bar.set_description("==>epoch:{:4d}, loss_D:{:10.5f}, loss_G:{:10.5f}".format(epoch, loss_D.cpu().item(), loss_G.cpu().item()))
        writer.add_scalars(main_tag="loss", tag_scalar_dict={
            "loss_d":loss_D.cpu().item(),
            "loss_g":loss_G.cpu().item()
        }, global_step=batch_cnt)
        if batch_cnt%100 == 0:
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
            
            #torch.save(model_G.state_dict(), f=os.path.join(args.root, model_dir, "./model/model_G_{}.pytorch".format(save_idx)))
            #torch.save(model_D.state_dict(), f=os.path.join(args.root, model_dir, "./model/model_D_{}.pytorch".format(save_idx)))
            save_idx += 1


if __name__=="__main__":
    for epoch in range(0, args.epoches):
        train(epoch)


        