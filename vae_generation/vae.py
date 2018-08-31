import argparse
import os
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
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epoches", type=int, default=100, help="set the training epoches. default(10)")
parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of encoding data.(default 20)")
parser.add_argument("--root", type=str, default="/home/jiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    print("cuda in available, use cuda")
else:
    print("cuda in not available, use cpu")

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.root + "../data/mnist", train=True, download=False, 
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ])
    ),
    batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.root + "../data/mnist", train=False, 
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ])),
    batch_size=args.batch_size, shuffle=True
)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2_mu = nn.Linear(400, self.hidden_dim)
        self.fc2_log_var = nn.Linear(400, self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim,400)
        self.fc4 = nn.Linear(400, 28*28)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, input_data):
        x = self.relu(self.fc1(input_data.view(-1, 1*28*28)))
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

    def decode(self, z):
        x = self.relu(self.fc3(z))
        x = self.fc4(x)
        return self.sigmoid(x).view(-1, 1, 28, 28)

    def forward(self, input_data):
        mu, log_var = self.encode(input_data)
        z = self.reparameter(mu, log_var)
        rec = self.decode(z)
        return rec, mu, log_var


model = VAE()
if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def loss_func(input_data, rec, mu, log_var):
    """
    loss function for VAE
    input: x, rec_x, mu, log_var
    return: loss for VAE
    """
    BCE = F.binary_cross_entropy(rec.view(-1, 28*28), input_data.view(-1, 28*28), reduce=False).sum(dim=1).mean()  #why the size_average is False
    # -0.5*sum(1+log(sigma^2)-mu^2-sigma^2), how this func is constructed
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1))
    return BCE, KLD

def train(epoch):
    model.train()
    train_loss = 0
    loader_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        data = data.requires_grad_(True)
        if args.cuda:
            data = data.cuda()

        data_rec, mu, log_var = model.forward(data)
        loss_BCE, loss_KLD = loss_func(input_data=data.detach().requires_grad_(False), rec=data_rec, mu=mu, log_var=log_var)
        loss = loss_BCE + loss_KLD

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
        data = data.requires_grad_(False)
        if args.cuda:
            data = data.cuda()

        data_rec, mu, log_var = model(data)
        loss_BCE, loss_KLD = loss_func(data, data_rec, mu, log_var)
        loss = loss_BCE + loss_KLD

        test_loss += loss.cpu().item()
        
        loader_bar.set_description("[{:5s}] epoch:{:5d}, loss:{:10.5f}[{:10.5f},{:10.5f}]".format("test", epoch, 
            loss.cpu().item(), loss_BCE.cpu().item(), loss_KLD.cpu().item()
            ))

        if batch_idx == 0:
            n = min(8, args.batch_size)
            comparision = torch.cat([data[:n], data_rec.view_as(data)[:n]])
            if not os.path.exists(args.root+"./vae_generation/rec/"):
                os.system("mkdir {}".format(args.root+"./vae_generation/rec/"))

            torchvision.utils.save_image(tensor=comparision.view(-1,1,28,28).data.cpu(), 
                    filename=args.root+"./vae_generation/rec/comparision_{}.png".format(epoch), normalize=True, range=(0,1), nrow=8)

    print("[{:5s}] epoch:{:5d}, loss:{:10.5f}".format("test", epoch, test_loss/len(test_loader)))



if __name__ == "__main__":
    for epoch in range(args.epoches):
        train(epoch)
        test(epoch)
        #sample
        sampler = torch.randn(64, args.hidden_dim)
        sampler = sampler.requires_grad_(False)
        if args.cuda:
            sampler = sampler.cuda()
        gene_sampler = model.decode(sampler)
        
        if not os.path.exists(args.root+"./vae_generation/rec/"):
                os.system("mkdir {}".format(args.root+"./vae_generation/rec/"))
                
        torchvision.utils.save_image(tensor=gene_sampler.data.cpu(), 
                filename=args.root+"./vae_generation/rec/gene_sampler_{}.png".format(epoch), normalize=True, range=(0,1), nrow=8)
        