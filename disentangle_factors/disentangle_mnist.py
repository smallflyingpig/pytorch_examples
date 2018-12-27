import os
import sys
import argparse
import tqdm

import torch
import numpy as np
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets import MNIST

from torchvision import datasets, transforms


parser = argparse.ArgumentParser("mnist disentangle")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda")
parser.add_argument("--alpha", type=float, default=1.0, help="coeffi for mse loss")
parser.add_argument("--beta", type=float, default=1.0, help="coeffi for KL loss")
parser.add_argument("--log_dir", type=str, default="./disentangle_factors/log")
parser.add_argument("--data_root", type=str, default="../data/mnist", help="data path for mnist")
parser.add_argument("--s_dim", type=int, default=16, help="dim for specified factors")
parser.add_argument("--z_dim", type=int, default=16, help="dim for unspecified factors")
parser.add_argument("--class_num", type=int, default=10, help="class num for the dataset, 10 for mnist")
parser.add_argument("--lr_G", type=float, default=1e-3, help="learning rate")
parser.add_argument("--lr_D", type=float, default=1e-4, help="learning rate")
parser.add_argument("--epoch", type=int, default=100, help="total epoch")

args, _ = parser.parse_known_args()

if not args.no_cuda and torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False

writer = SummaryWriter(log_dir=args.log_dir)

class MNIST_Triplet(MNIST):
    def __init__(self, root, train=True, transforms=None, target_transform=None, download=False):
        super(MNIST_Triplet, self).__init__(root, train, transforms, target_transform, download)
        self.data = self.train_data if train else self.test_data
        self.labels = self.train_labels if train else self.test_labels

        self.labels = self.labels.long()
        self.label_list = self.labels.unique().tolist()
        
        self.data_idx_class_aware = {label:np.argwhere(self.labels==label)[0].numpy() for label in self.label_list}

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        data_same_idx = np.random.choice(self.data_idx_class_aware[target.item()], size=1)[0]
        label_list_diff = self.label_list.copy()
        label_list_diff.remove(target.item())
        data_diff_class_label = np.random.choice(label_list_diff, size=1)[0]
        data_diff_idx = np.random.choice(self.data_idx_class_aware[data_diff_class_label], size=1)[0]
        img_same, target_same = self.data[data_same_idx], self.labels[data_same_idx]
        img_diff, target_diff = self.data[data_diff_idx], self.labels[data_diff_idx]
        img, img_same, img_diff = (img.float()-128)/128, (img_same.float()-128)/128, (img_diff.float()-128)/128
        
        return (img, target), (img_same, target_same), (img_diff, target_diff)

    def class_aware_sample(self):
        sample_idxs = [np.random.choice(self.data_idx_class_aware[class_key], size=1)[0] for class_key in self.data_idx_class_aware.keys()]
        imgs = [self.data[idx] for idx in sample_idxs]
        targets = [self.labels[idx] for idx in sample_idxs]
        imgs, targets = torch.stack(imgs), torch.stack(targets)
        imgs = (imgs.float()-128)/128
        return imgs, targets

    

class ConvLayer2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3,1,1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DisentangleNet(nn.Module):
    def __init__(self, s_dim, z_dim, input_size=28, input_channel=1):
        super(DisentangleNet, self).__init__()
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.input_size = input_size
        self.input_channel = input_channel
        self.base_dim = 16
        self.enc = nn.Sequential(
            # (1,28,28)
            ConvLayer2d(self.input_channel, self.base_dim),
            nn.MaxPool2d(2,2,1),
            # (8,14,14)
            ConvLayer2d(self.base_dim,self.base_dim*2),
            nn.MaxPool2d(2,2,1),
            # (16, 7, 7)
            ConvLayer2d(self.base_dim*2,self.base_dim*4),
            nn.MaxPool2d(3,2,1),
            # (32, 3,3)
            ConvLayer2d(self.base_dim*4,self.base_dim*8),
            nn.MaxPool2d(2,2,1),
            nn.Conv2d(self.base_dim*8, self.s_dim+self.z_dim+self.z_dim, 2,2,0),
            # nn.Conv2d(64, self.s_dim+self.z_dim+self.z_dim,1,1,0, bias=True)
            # (20,1,1)
        )
        self.dec1 = nn.Sequential(
            # (20,1,1)
            nn.ConvTranspose2d(self.s_dim+self.z_dim, self.base_dim*8, 3,1,0, bias=False),
            nn.BatchNorm2d(self.base_dim*8),
            nn.ReLU(inplace=True),
            # (32, 3, 3)
            ConvLayer2d(self.base_dim*8, self.base_dim*4),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (16, 6, 6)
            ConvLayer2d(self.base_dim*4, self.base_dim*2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (8, 12, 12)
            ConvLayer2d(self.base_dim*2, self.base_dim),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # (4, 24, 24),
            nn.ConvTranspose2d(self.base_dim, 1, self.input_size-24+1,1,0, bias=True),
            # 
            nn.Tanh()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.s_dim+self.z_dim, self.base_dim*8, 3, 1, 0, bias=False),
            nn.BatchNorm2d(self.base_dim*8),
            nn.ReLU(True),
            # batch x 512 x 3 x 3 --> batch x 256 x 6 x 6
            nn.ConvTranspose2d(self.base_dim*8, self.base_dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_dim*4),
            nn.ReLU(True),
            # batch x 256 x 6 x 6 --> batch x 128 x 12 x 12
            nn.ConvTranspose2d(self.base_dim*4, self.base_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_dim*2),
            nn.ReLU(True),
            # batch x 128 x 12 x 12 --> batch x  64 x 24 x 24
            nn.ConvTranspose2d(self.base_dim*2, self.base_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.base_dim),
            nn.ReLU(True),
            # batch x 64 x 24 x 24 --> batch x channel x 28 x 28
            nn.ConvTranspose2d(self.base_dim, self.input_channel, 5, 1, 0),
            nn.Tanh()
        )

    def reparameter(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var*0.5)
            eps = (std.data.new(std.size()).normal_())
             
            return mu + eps.mul(std).to(mu.device)
        else:
            return mu

    def encode(self, x):
        hidden = self.enc(x).squeeze()
        s, mu, logvar = hidden[:, :self.s_dim], hidden[:, self.s_dim:self.s_dim+self.z_dim], hidden[:, -self.z_dim:]
        return s, mu, logvar

    def decode(self, s, z):
        x = self.dec(torch.cat([z,s], 1).unsqueeze(-1).unsqueeze(-1))
        return x

    def forward(self, x):
        x0, x_same, x_diff = x
        if not len(x0.shape)==len(x_same.shape)==len(x_diff.shape)==4:
            print("input shape error")
            return None
        hid_0, hid_same, hid_diff = self.enc(x0).squeeze(), self.enc(x_same).squeeze(), self.enc(x_diff).squeeze()
        s0, mu0, logvar0 = hid_0[:, :self.s_dim], hid_0[:, self.s_dim:self.s_dim+self.z_dim], hid_0[:, -self.z_dim:]
        s_same, mu_same, logvar_same = hid_same[:, :self.s_dim], hid_same[:, self.s_dim:self.s_dim+self.z_dim], hid_same[:, -self.z_dim:]
        s_diff, mu_diff, logvar_diff = hid_diff[:, :self.s_dim], hid_diff[:, self.s_dim:self.s_dim+self.z_dim], hid_diff[:, -self.z_dim:]

        z0, z_same, z_diff = self.reparameter(mu0, logvar0), self.reparameter(mu_same, logvar_same), self.reparameter(mu_diff, logvar_diff)
        z_noise_diff = torch.randn_like(z0).to(z0.device)

        x0_0, x0_same, x0_diff, x_noise_diff = torch.cat([z0, s0], 1).unsqueeze(-1).unsqueeze(-1), torch.cat([z0, s_same], 1).unsqueeze(-1).unsqueeze(-1), \
            torch.cat([z0, s_diff], 1).unsqueeze(-1).unsqueeze(-1), torch.cat([z_noise_diff, s_diff], 1).unsqueeze(-1).unsqueeze(-1)
        x0_0, x0_same, x0_diff = self.dec(x0_0), self.dec(x0_same), self.dec(x0_diff)
        
        x_noise_diff = self.dec(x_noise_diff)

        return x0_0, x0_same, x0_diff, x_noise_diff, (mu0, logvar0), (mu_same, logvar_same), (mu_diff, logvar_diff)


class ConditionGAN(nn.Module):
    def __init__(self, condition_dim, input_channel=1, input_size=28):
        super(ConditionGAN, self).__init__()
        self.input_channel = input_channel
        self.input_size = input_size
        self.condition_dim = condition_dim
        self.base_dim = 16
        self.enc = nn.Sequential(
            # (1,28,28)
            ConvLayer2d(self.input_channel, self.base_dim),
            nn.MaxPool2d(2,2,1),
            # (8, 14, 14)
            ConvLayer2d(self.base_dim, self.base_dim*2),
            nn.MaxPool2d(2,2,1),
            # (16, 7, 7)
            ConvLayer2d(self.base_dim*2, self.base_dim*4),
            nn.MaxPool2d(3,2,1),
            # (32, 3, 3),
            # nn.AvgPool2d(self.input_size//8),
            # (1, 1, 1)
        )
        self.logit_layer = nn.Sequential(
            ConvLayer2d(self.base_dim*4+self.condition_dim, self.base_dim*4),
            nn.AvgPool2d(4),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(self.base_dim*4, 1, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Sigmoid()
        )

        self.logit_layer1 = nn.Sequential(
            ConvLayer2d(self.base_dim*4+self.condition_dim, self.base_dim*4),
            nn.AvgPool2d(4),
        )
        self.out_layer1 = nn.Sequential(
            nn.Linear(self.base_dim*4, 1, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Sigmoid()
        )


    def forward(self, x, c):
        if not len(x.shape)==4:
            print("input shape error")
            return None
        x = self.enc(x)
        x_c = torch.cat([x, c.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4,4)], dim=1)
        pred = self.out_layer(self.logit_layer(x_c).squeeze())
        return pred.squeeze()


def loss_func_KLD(mu, log_var):
    """
    loss function for VAE
    input: x, rec_x, mu, log_var
    return: loss for VAE
    """
    batch_size = mu.shape[0]
    #BCE = F.binary_cross_entropy(rec.view(batch_size, -1), input_data.view(batch_size, -1), reduce=False).sum(dim=1).mean()  #why the size_average is False
    # -0.5*sum(1+log(sigma^2)-mu^2-sigma^2), how this func is constructed
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1))
    return KLD

def loss_func_BCE(input, target):
    error = F.binary_cross_entropy(input, target)
    return error

def loss_func_MSE(data_rec, data):
    batch_size = data.shape[0]
    error = F.mse_loss(data_rec.view(batch_size, -1), data.view(batch_size, -1)).mean()
    return error

def loss_func_L1(data_rec, data):
    batch_size = data.shape[0]
    error = F.l1_loss(data_rec.view(batch_size, -1), data.view(batch_size, -1)).mean()
    return error

batch_cnt = 0
def train_once(model, optimizer, dataloader, epoch):
    global batch_cnt
    model_disentangle, model_discriminator = model
    model_disentangle.train()
    model_discriminator.train()
    optimizer_disentangle, optimizer_discriminator = optimizer
    loss_epoch_D, loss_epoch_G = 0, 0
    bar = tqdm.tqdm(dataloader)
    for idx, batch_data in enumerate(bar):
        batch_cnt += 1
        (x0, label0), (x_same, label_same), (x_diff, label_diff) = batch_data
        x0, x_same, x_diff = x0.float().unsqueeze(1), x_same.float().unsqueeze(1), x_diff.float().unsqueeze(1)
        batch_size = len(x0)
        label_real = torch.FloatTensor(batch_size).fill_(1)
        label_fake = torch.FloatTensor(batch_size).fill_(0)
        label_diff_onehot = torch.eye(args.class_num)[label_diff].reshape(batch_size, args.class_num)
        label_same_onehot = torch.eye(args.class_num)[label_same].reshape(batch_size, args.class_num)

        x0, x_same, x_diff = x0.requires_grad_(), x_same.requires_grad_(), x_diff.requires_grad_()
        if args.cuda:
            x0, label0, x_same, label_same, x_diff, label_diff, label_real, label_fake = \
                x0.cuda(), label0.cuda(), x_same.cuda(), label_same.cuda(), x_diff.cuda(), label_diff.cuda(), label_real.cuda(), label_fake.cuda()
            label_diff_onehot, label_same_onehot = label_diff_onehot.cuda(), label_same_onehot.cuda()
        # forward
        output = model_disentangle.forward((x0, x_same, x_diff))
        x0_0, x0_same, x0_diff, x_noise_diff, (mu0, logvar0), (mu_same, logvar_same), (mu_diff, logvar_diff) = output


        # update discriminator
        pred_real_diff = model_discriminator(x_diff, label_diff_onehot)
        pred_fake_diff = model_discriminator(x0_diff.detach(), label_diff_onehot)
        pred_real_0 = model_discriminator(x0, label_same_onehot)
        pred_fake_0 = model_discriminator(x0_0.detach(), label_same_onehot)
        pred_real_same = model_discriminator(x_same, label_same_onehot)
        pred_fake_same = model_discriminator(x0_same.detach(), label_same_onehot)
        # pred_fake1 = model_discriminator(x_noise_diff.detach(), label_diff_onehot)

        loss_real = (loss_func_MSE(pred_real_diff, label_real) + loss_func_MSE(pred_real_0, label_real) + loss_func_MSE(pred_real_same, label_real))/3 #loss_func_BCE(pred_real, label_real)  #-pred_real.mean()  #
        loss_fake = (loss_func_MSE(pred_fake_diff, label_fake) + loss_func_MSE(pred_fake_0, label_fake) + loss_func_MSE(pred_fake_same, label_fake))/3 #(loss_func_BCE(pred_fake, label_fake) + loss_func_BCE(pred_fake1, label_fake))/2  #(pred_fake + pred_fake1).mean()/2  #
        loss_dis =  (loss_real + loss_fake)/2

        model_disentangle.zero_grad()
        model_discriminator.zero_grad()
        loss_dis.backward()
        # clip the param
        # for p in model_discriminator.parameters():
        #     p.data.clamp_(-0.02, 0.02)

        optimizer_discriminator.step()

        # update disentangle net
        output = model_disentangle.forward((x0, x_same, x_diff))
        x0_0, x0_same, x0_diff, x_noise_diff, (mu0, logvar0), (mu_same, logvar_same), (mu_diff, logvar_diff) = output

        pred0_0 = model_discriminator(x0_0, label_same_onehot)
        pred0_same = model_discriminator(x0_same, label_same_onehot)
        pred0_diff = model_discriminator(x0_diff, label_diff_onehot)
        pred_noise_diff = model_discriminator(x_noise_diff, label_diff_onehot)
        # loss
        loss_mse = (loss_func_L1(x0_0, x0) + loss_func_L1(x0_same, x0))/2
        loss_bce = (loss_func_MSE(pred0_diff, label_real) + loss_func_MSE(pred_noise_diff, label_real) + loss_func_MSE(pred0_0, label_real) + loss_func_MSE(pred0_same, label_real))/4 #(loss_func_BCE(pred0_diff, label_real) + loss_func_BCE(pred_noise_diff, label_real))/2  #-(pred0_diff+pred_noise_diff).mean()/2  #
        loss_kld = (loss_func_KLD(mu0, logvar0) + loss_func_KLD(mu_same, logvar_same) + loss_func_KLD(mu_diff, logvar_diff))/3
        loss_all = args.alpha * loss_mse + loss_bce + args.beta * loss_kld

        model_discriminator.zero_grad()
        model_disentangle.zero_grad()
        loss_all.backward()
        optimizer_disentangle.step()

        writer.add_scalars("loss_D", {
            "loss_real":loss_real.item(),
            "loss_fake":loss_fake.item(),
            "loss_all":loss_dis.item()
        }, batch_cnt)
        writer.add_scalars("loss_G", {
            "loss_mse": loss_mse.item(),
            "loss_bce": loss_bce.item(),
            "loss_kld": loss_kld.item(),
            "loss_all": loss_all.item()
        }, batch_cnt)
        if batch_cnt % 1000 == 0:
            writer.add_image(tag="x0", img_tensor=torchvision.utils.make_grid(x0, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="x0_0", img_tensor=torchvision.utils.make_grid(x0_0, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="x0_same", img_tensor=torchvision.utils.make_grid(x0_same, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="x_diff", img_tensor=torchvision.utils.make_grid(x_diff, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="x0_diff", img_tensor=torchvision.utils.make_grid(x0_diff, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="x_noise_diff", img_tensor=torchvision.utils.make_grid(x_noise_diff, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)

        loss_epoch_D += loss_dis.item()
        loss_epoch_G += loss_all.item()
        bar.set_description("[train] epoch:{:5d} batch:{:5d} loss D:{:6.3f} loss G:{:6.3f}".format(epoch, idx, loss_dis.item(), loss_all.item()))
    bar.close()
    loss_epoch_D /= len(dataloader)
    loss_epoch_G /= len(dataloader)

    print("[train] epoch:{:5d} loss D:{:6.3f} loss G:{:6.3f}".format(epoch, loss_epoch_D, loss_epoch_G))
    return loss_epoch_D, loss_epoch_G
    


def test(model_disentangle, dataloader, epoch):
    model_disentangle.eval()
    data = dataloader.dataset.class_aware_sample()
    assert(data is not None)
    images, targets = data
    images, targets = images.float().unsqueeze(1), targets.float().unsqueeze(1)
    if args.cuda:
        images, targets = images.cuda(), targets.cuda()
    s, mu, logvar = model_disentangle.encode(images)

    images_all = torch.FloatTensor(args.class_num+1, args.class_num+1, 28, 28).fill_(1)
    
    s_all = []
    z_all = []
    for idx1 in range(args.class_num):
        for idx2 in range(args.class_num):
            s_all.append(s[idx1])
            z_all.append(mu[idx2])
    s_all, z_all = torch.stack(s_all), torch.stack(z_all)

    img_rec_all = model_disentangle.decode(s_all, z_all).reshape(args.class_num, args.class_num, 28, 28)

    images_all[1:, 1:] = img_rec_all.cpu()
    images_all[0, 1:] = images.cpu().squeeze()
    images_all[1:, 0] = images.cpu().squeeze()
    images_all = images_all.reshape((args.class_num+1)*(args.class_num+1), 1, 28, 28)
    writer.add_image(tag="sample", img_tensor=torchvision.utils.make_grid(images_all, nrow=args.class_num+1, padding=0, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)


def main():
    train_dataset = MNIST_Triplet(root=args.data_root, train=True)
    test_dataset = MNIST_Triplet(root=args.data_root, train=False)
    model_disentangle = DisentangleNet(s_dim=args.s_dim, z_dim=args.z_dim)
    model_discriminator = ConditionGAN(condition_dim=args.class_num)

    if args.cuda:
        model_disentangle, model_discriminator = model_disentangle.cuda(), model_discriminator.cuda()
    optimizer_disentangle = torch.optim.Adam(model_disentangle.parameters(), lr=args.lr_G)
    optimizer_discrimminator = torch.optim.Adam(model_discriminator.parameters(), lr=args.lr_D)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset)
    test(model_disentangle, test_dataloader, 0)
    for epoch_idx in range(args.epoch):
        train_once(model=(model_disentangle, model_discriminator), optimizer=(optimizer_disentangle, optimizer_discrimminator), dataloader=train_dataloader, epoch=epoch_idx)
        test(model_disentangle, test_dataloader, epoch_idx)

    


if __name__=="__main__":
    main()





        


    





