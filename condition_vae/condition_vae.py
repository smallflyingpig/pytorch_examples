import argparse
import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description="VAE mnist examples")
parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training.(default 64)")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda training")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate, default 1e-4")
parser.add_argument("--epoches", type=int, default=100, help="set the training epoches. default(10)")
parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of encoding data.(default 20)")
parser.add_argument("--root", type=str, default="/home/jiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")
parser.add_argument("--model", type=str, default="mlp", 
                    help="model type for VAE encoder and decoder, mlp or conv, decault mlp")
parser.add_argument("--img_size", type=int, default=32, 
                    help="image size the image will resize to, default 32")                    
parser.add_argument("--KLD_coeff", type=float, default=5, 
                    help="loss coeff of KLD, default 5")  
parser.add_argument("--class_num", type=int, default=10, 
                    help="class number of the data, default 10")      
parser.add_argument("--condition_dim", type=int, default=10, 
                    help="condition dim to concate to the noie, default 10")                                                          

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_dir = "condition_vae"
rec_dir = os.path.join(args.root, model_dir, "./rec")

if args.cuda:
    print("cuda in available, use cuda")
else:
    print("cuda in not available, use cpu")

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)

param = {"num_workers":4, "pin_memory":True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.root + "../data/mnist", train=True, download=False, 
                    transform=transforms.Compose([
                    transforms.Resize(size=(args.img_size)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ]), 
    ),
    batch_size=args.batch_size, shuffle=True, **param
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.root + "../data/mnist", train=False, 
                    transform=transforms.Compose([
                    transforms.Resize(size=(args.img_size)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ])),
    batch_size=args.batch_size, shuffle=True, **param
)

class VAE_conv(nn.Module):
    def __init__(self, img_size, img_channel=1, condition_dim=10):
        super(VAE_conv, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.condition_dim = condition_dim
        self.img_size = img_size
        self.img_channel = img_channel
        self.ref_dim = 32

        self.encoder = nn.Sequential(
            # img_size --> img_size//2
            nn.Conv2d(self.img_channel, self.ref_dim, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim),
            nn.LeakyReLU(0.2, True),
            # img_size//2 --> img_size//4
            nn.Conv2d(self.ref_dim, self.ref_dim*2, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*2),
            nn.LeakyReLU(0.2, True),
            # img_size//4 --> img_size//8
            nn.Conv2d(self.ref_dim*2, self.ref_dim*4, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*4),
            nn.LeakyReLU(0.2, True),
        )
        self.project_dim = self.ref_dim*4 * self.img_size[0]//8 * self.img_size[1]//8
        self.project_layer = nn.Sequential(
            nn.Linear(self.ref_dim*4 * self.img_size[0]//8 * self.img_size[1]//8, self.hidden_dim*2, bias=True),
        )
        self.decoder = nn.Sequential(
            # hidden_dim x 1 x 1 --> ref_dim*4 x img_size//8 x img_size//8
            nn.ConvTranspose2d(self.hidden_dim+self.condition_dim, self.ref_dim*4, self.img_size[0]//8,1,0, bias=False),
            nn.BatchNorm2d(self.ref_dim*4),
            nn.ReLU(True),
            # ref_dim*4 x img_size//8 x img_size//8 --> ref_dim*2 x img_size//4 x img_size//4
            nn.ConvTranspose2d(self.ref_dim*4, self.ref_dim*2, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*2),
            nn.ReLU(True),
            # ref_dim*2 x img_size//4 x img_size//4 --> ref_dim*1 x img_size//2 x img_size//2
            nn.ConvTranspose2d(self.ref_dim*2, self.ref_dim*1, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ref_dim, self.img_channel, 4,2,1, bias=True),
            nn.Sigmoid()
        )

    def encode(self, input_data):
        output = self.encoder(input_data)
        output = self.project_layer(output.view(-1, self.project_dim))
        mu, log_var = output[:,:self.hidden_dim], output[:,self.hidden_dim:]
        return mu, log_var

    def reparameter(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var*0.5)
            eps = Variable(std.data.new(std.size()).normal_())
            if args.cuda:
                eps.cuda()
            return mu + eps.mul(std)
        else:
            return mu

    def decode(self, z, c):
        z_c = torch.cat([z,c],1)
        x = self.decoder(z_c.unsqueeze(2).unsqueeze(3))
        return x.view(-1, 1, self.img_size[0], self.img_size[1])

    def forward(self, input_data, condition):
        mu, log_var = self.encode(input_data)
        z = self.reparameter(mu, log_var)
        rec = self.decode(z, condition)
        return rec, mu, log_var


class VAE_mlp(nn.Module):
    def __init__(self, img_size, condition_dim=10):
        super(VAE_mlp, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.condition_dim = condition_dim
        self.img_size = img_size
        self.img_size_flatten = self.img_size[0]*self.img_size[1]
        self.fc1 = nn.Linear(self.img_size_flatten, 400)
        self.fc2_mu = nn.Linear(400, self.hidden_dim)
        self.fc2_log_var = nn.Linear(400, self.hidden_dim)

        # for decoder
        self.fc3 = nn.Linear(self.hidden_dim+self.condition_dim,400)
        self.fc4 = nn.Linear(400, self.img_size_flatten)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, input_data):
        x = self.relu(self.fc1(input_data.view(-1, self.img_size_flatten)))
        mu = self.fc2_mu(x)
        log_var = self.fc2_log_var(x)
        return mu, log_var

    def reparameter(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var*0.5)
            eps = Variable(std.data.new(std.size()).normal_())
            if args.cuda:
                eps.cuda()
            return mu + eps.mul(std)
        else:
            return mu

    def decode(self, z, c):
        z_c = torch.cat([z, c], 1)
        x = self.relu(self.fc3(z_c))
        x = self.fc4(x)
        return self.sigmoid(x).view(-1, 1, self.img_size[0], self.img_size[1])

    def forward(self, input_data, condition):
        mu, log_var = self.encode(input_data)
        z = self.reparameter(mu, log_var)
        rec = self.decode(z, condition)
        return rec, mu, log_var


if args.model == "mlp":
    model = VAE_mlp(img_size=(args.img_size, args.img_size), condition_dim=args.condition_dim)
elif args.model == "conv":
    model = VAE_conv(img_size=(args.img_size, args.img_size), condition_dim=args.condition_dim)
else:
    model = VAE_mlp(img_size=(args.img_size, args.img_size), condition_dim=args.condition_dim)

if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def loss_func(input_data, rec, mu, log_var):
    """
    loss function for VAE
    input: x, rec_x, mu, log_var
    return: loss for VAE
    """
    batch_size = input_data.shape[0]
    BCE = F.binary_cross_entropy(rec.view(batch_size, -1), input_data.view(batch_size, -1), reduce=False).sum(dim=1).mean()  #why the size_average is False
    # -0.5*sum(1+log(sigma^2)-mu^2-sigma^2), how this func is constructed
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1))
    return BCE, KLD

def train(epoch):
    model.train()
    train_loss = 0
    loader_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        batch_size = data.shape[0]
        data = data.requires_grad_(True)
        
        condition = torch.eye(args.class_num)[label].reshape(batch_size, args.class_num)

        if args.cuda:
            data, label, condition = data.cuda(), label.cuda(), condition.cuda()
        
        data_rec, mu, log_var = model.forward(data, condition)
        loss_BCE, loss_KLD = loss_func(input_data=data.detach().requires_grad_(False), rec=data_rec, mu=mu, log_var=log_var)
        loss = loss_BCE + args.KLD_coeff*loss_KLD

        model.zero_grad()
        loss.backward()
        
        optimizer.step()

        train_loss += loss.cpu().item()
        loader_bar.set_description("[{:5s}] epoch:{:5d}, loss:{:10.5f}[{:10.5f},{:10.5f}]".format("train", epoch, 
            loss.cpu().item(), loss_BCE.cpu().item(), loss_KLD.cpu().item()
            ))
            
    print("[{:5s}] epoch:{:5d}, loss:{:10.5f}".format("train", epoch, train_loss/(len(train_loader))))



def test(epoch):
    model.eval()
    test_loss = 0
    loader_bar = tqdm.tqdm(test_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        batch_size = data.shape[0]
        data = data.requires_grad_(False)
        condition = torch.eye(args.class_num)[label].reshape(batch_size, args.class_num)
        if args.cuda:
            data, condition = data.cuda(), condition.cuda()

        data_rec, mu, log_var = model(data, condition)
        loss_BCE, loss_KLD = loss_func(data, data_rec, mu, log_var)
        loss = loss_BCE + args.KLD_coeff*loss_KLD

        test_loss += loss.cpu().item()
        
        loader_bar.set_description("[{:5s}] epoch:{:5d}, loss:{:10.5f}[{:10.5f},{:10.5f}]".format("test", epoch, 
            loss.cpu().item(), loss_BCE.cpu().item(), loss_KLD.cpu().item()
            ))

        if batch_idx == 0:
            n = min(8, args.batch_size)
            comparision = torch.cat([data[:n], data_rec.view_as(data)[:n]])
            if not os.path.exists(rec_dir):
                os.system("mkdir {}".format(rec_dir))

            torchvision.utils.save_image(tensor=comparision.view(-1,1,args.img_size,args.img_size).data.cpu(), 
                    filename=os.path.join(rec_dir, "./comparision_{}.png".format(epoch)), normalize=True, range=(0,1), nrow=8)

    print("[{:5s}] epoch:{:5d}, loss:{:10.5f}".format("test", epoch, test_loss/len(test_loader)))


def sample(epoch):
    condition_idx = np.array(list(range(args.class_num*args.class_num)))
    condition_idx = condition_idx%args.class_num
    condition = np.eye(args.class_num)[condition_idx.astype(np.int32)]
    condition = torch.Tensor(condition)
    sampler = torch.randn(condition.shape[0], args.hidden_dim)
    sampler = sampler.requires_grad_(False)
    if args.cuda:
        sampler, condition = sampler.cuda(), condition.cuda()

    gene_sampler = model.decode(sampler, condition)
    
    if not os.path.exists(rec_dir):
            os.system("mkdir {}".format(rec_dir))
            
    torchvision.utils.save_image(tensor=gene_sampler.data.cpu(), 
            filename=os.path.join(rec_dir, "./gene_sampler_{}.png".format(epoch)), normalize=True, range=(0,1), nrow=args.class_num)


if __name__ == "__main__":
    for epoch in range(args.epoches):
        train(epoch)
        test(epoch)
        sample(epoch)
        
        