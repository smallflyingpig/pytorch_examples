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
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training data. default 64")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda use")
parser.add_argument("--epoches", type=int, default=50, help="training epoches, default 10")
parser.add_argument("--hidden_dim", type=int, default=1024, help="hidden dimension of gan. (default 1024, 3 layers total)")
parser.add_argument("--z_dim", type=int, default=64, help="dimension of z. default 2")

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

model_dir = "./began"
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
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.noise_dim, 256),
            nn.BatchNorm1d(256)
        )
        self.decoder = nn.Sequential(
            # 256 x 1 x 1 --> 128 x 7 x 7
            nn.ConvTranspose2d(256, 128, 7, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 7 x 7 --> 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 14 x 14 --> 1 x 7 x 7
            nn.ConvTranspose2d(64, self.channel, 4, 2, 1),
        )
        self.output = nn.Tanh()
    
    def forward(self, z):
        batch_size = z.shape[0]
        z = z.view(batch_size, -1)
        x = self.decoder(
            self.decoder_linear(z).unsqueeze(2).unsqueeze(3)
        )
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    """
    discriminator for GAN, autoencoder style
    """
    def __init__(self, img_dim=28, channel=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.channel = channel
        self.img_dim = img_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            # 1 x 28 x 28 --> 64 x 14 x 14
            nn.Conv2d(self.channel, 64, 3, 2, 1),
            nn.ReLU(True),
            # 64 x 14 x 14 --> 128 x 7 x 7
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(True),
            # 128 x 7 x 7 --> 256 x 1 x 1
            nn.Conv2d(128, 256, 7, 1, 0),
            nn.ReLU(True),
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(256, self.hidden_dim)
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(self.hidden_dim, 256)
        )
        self.decoder = nn.Sequential(
            # 256 x 1 x 1 --> 128 x 7 x 7
            nn.ConvTranspose2d(256, 128, 7, 1, 0),
            nn.ReLU(True),
            # 128 x 7 x 7 --> 64 x 14 x 14
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            # 64 x 14 x 14 --> 1 x 7 x 7
            nn.ConvTranspose2d(64, self.channel, 4, 2, 1),
        )


    def forward(self, z):
        """
        input: z (batch_size x 1 x 28 x 28)
        output: pred (batch_size x 1)
        """
        batch_size = z.shape[0]
        h_hidden = self.encoder_linear(
            self.encoder(z).view(batch_size, -1)
        )
        rec = self.decoder(
            self.decoder_linear(h_hidden).unsqueeze(2).unsqueeze(3)
        )
        return rec

def loss_func(input, target):
    error = F.l1_loss(input, target)
    return error


model_G = Generator(noise_dim=args.z_dim)
model_D = Discriminator(hidden_dim=args.z_dim)

if args.cuda:
    model_G.cuda()
    model_D.cuda()

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=1e-5)# torch.optim.SGD(model_G.parameters(), lr=1e-3, momentum=0.9)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-5)# torch.optim.SGD(model_D.parameters(), lr=1e-4, momentum=0.9)

save_idx = 0
batch_cnt = 0
def train(epoch):
    """
    """
    global batch_cnt
    global save_idx

    model_D.train()
    model_G.train()
    param_k = 0
    param_gamma = 1
    param_lambda = 0.001
    for batch_idx, (data, label) in enumerate(train_loader):
        batch_cnt += 1
        batch_size = data.size(0)
        #real data
        data_real = data
        data_real = Variable(data_real)

        if args.cuda:
            data_real = data_real.cuda()
        #fake data
        z = torch.rand(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_(True)
        if args.cuda:
            z = z.cuda()

        data_fake = model_G.forward(z)

        #update D
        model_D.zero_grad()
        pred_D_real = model_D.forward(data_real)
        pred_D_fake = model_D.forward(data_fake.detach())
        loss_D_real = loss_func(pred_D_real, data_real)
        loss_D_fake = loss_func(pred_D_fake, data_fake.detach())
        # no log here
        loss_D = loss_D_real - param_k*loss_D_fake

        loss_D.backward()
        optimizer_D.step()

        #update G
        z = torch.rand(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_(False)
        if args.cuda:
            z = z.cuda()
        data_fake = model_G.forward(z)
        pred_G = model_D.forward(data_fake)
        model_G.zero_grad()
        # no log here
        loss_G = loss_func(pred_G, data_fake)
        loss_G.backward()
        optimizer_G.step()

        # updata k
        param_k += (param_lambda*(param_gamma*loss_D_real - loss_G)).item()

        writer.add_scalars(main_tag="loss", tag_scalar_dict={
            "loss_d":loss_D.cpu().item(),
            "loss_g":loss_G.cpu().item(),
            "loss_d_real":loss_D_real.cpu().item(),
            "loss_d_fake":loss_D_fake.cpu().item(),
            "param_k":param_k
        }, global_step=batch_cnt)
        if batch_idx%100 == 0:
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


        