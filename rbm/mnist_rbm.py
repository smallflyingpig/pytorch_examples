# reference: learning deep architecture for AI, Yoshua bengio, p26
# https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch

import argparse
import os
import sys
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np

parser = argparse.ArgumentParser("restricted boltzmann machine")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate for the training, default 0.001")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable the cuda")
parser.add_argument("--epoch", type=int, default=100, help="total epoch for the training")
parser.add_argument("--sample_folder", type=str, default="./rbm/samples", help="folder for samples data")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training")
parser.add_argument("--no_grad", action="store_true", default=False, help="disable the backward")
parser.add_argument("--nidden_dim", type=int, default=100, help="hidden dim, default 100")

args, _ = parser.parse_known_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

class RBM(nn.Module):
    def __init__(self, v_dim, h_dim, k=1):
        super(RBM, self).__init__()
        self.v_dim, self.h_dim, self.k = v_dim, h_dim, k
        self.weight = nn.Parameter(torch.randn(h_dim, v_dim)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(v_dim))
        self.h_bias = nn.Parameter(torch.zeros(h_dim))

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - torch.rand(p.shape).to(p.device))) # np.random.binomial(np.ones_like(p.shape), p, p.size

    def v_to_h(self, v):
        Q_h = F.sigmoid(F.linear(v, self.weight, self.h_bias)) #h_dim * 1
        h = self.sample_from_p(Q_h)
        return Q_h, h

    def h_to_v(self, h):
        Q_v = F.sigmoid(F.linear(h, self.weight.t(), self.v_bias)) #v_dim * 1
        v = self.sample_from_p(Q_v)
        return Q_v, v

    def forward(self, v):
        Q_h1, h1 = self.v_to_h(v)
        h_ = h1
        for _ in range(self.k):
            Q_v_, v_ = self.h_to_v(h_)
            Q_h_, h_ = self.v_to_h(v_)
        return v, Q_h1, v_, Q_h_

    def free_energy(self, v):
        v_bias_item = v.mv(self.v_bias)
        wx_b = F.linear(v, self.weight, self.h_bias)
        h_item = wx_b.exp().add(1).log().sum(dim=1)
        return (-h_item - v_bias_item).mean()

    def update_param(self, v1, Q_h1, vk, Q_hk, lr):
        h1 = Q_h1.bernoulli()

        grad_W = (h1.unsqueeze(2)*v1.unsqueeze(1)-Q_hk.unsqueeze(2)*vk.unsqueeze(1)).mean(dim=0)
        grad_h_bias = (h1 - Q_hk).mean(dim=0)
        grad_v_bias = (v1 - vk).mean(dim=0)

        #update, optim slow due to no momentum
        self.weight = nn.Parameter(self.weight.data + lr * 10 * grad_W)
        self.weight = nn.Parameter(self.weight.data - self.weight.data * 1e-5)
        self.h_bias = nn.Parameter(self.h_bias.data + lr * 10 * grad_h_bias)
        self.v_bias = nn.Parameter(self.v_bias.data + lr * 10 * grad_v_bias)


def train(model, dataloader, optimizer, epoch):
    for epoch_idx in range(epoch):
        loss_epoch = 0
        batch_bar = tqdm.tqdm(dataloader)
        for (data, _) in batch_bar:
            if args.cuda:
                data = data.cuda()
            data = data.requires_grad_().view(-1, 784)
            sample_data = data.bernoulli()
            v1, Q_h1, vk, Q_hk = model.forward(sample_data)
            loss =  model.free_energy(v1) - model.free_energy(vk)
            if not args.no_grad:
                model.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                model.update_param(v1, Q_h1, vk, Q_hk, args.learning_rate)

            loss_epoch += loss.cpu().detach().item()
            batch_bar.set_description("epoch: {:03d} loss: {:5.3f}".format(epoch_idx, loss.cpu().item()))
        batch_bar.set_description("loss: {:5.3f}".format(loss_epoch))
        #test and save
        save_data = torch.cat((v1.cpu().detach(), vk.cpu().detach()))
        save_name = "test_{:05d}.jpg".format(epoch_idx)
        torchvision.utils.save_image(save_data.view(-1,1,28,28), os.path.join(args.sample_folder, save_name))


def main():
    model = RBM(784, 100, 1)
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    param = {"num_workers":4, "pin_memory": True} if args.cuda else {}
    train_dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST("./data/mnist", train=True, download=True, 
            transform=transformer),
        batch_size=args.batch_size, drop_last=True, shuffle=True, **param)
    
    train(model, train_dataloader, optimizer, args.epoch)


if __name__ == "__main__":
    main()

            





