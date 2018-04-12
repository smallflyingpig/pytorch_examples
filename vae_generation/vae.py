import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description="VAE mnist examples")
parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training.(default 64)")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda training")
parser.add_argument("--epoches", type=int, default=10, help="set the training epoches. default(10)")
parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of encoding data.(default 20)")
parser.add_argument("--root", type=str, default="/home/jiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.root + "../data/mnist", train=True, download=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.root + "../data/mnist", train=False, transform=transforms.ToTensor()),
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
            eps = Variable(torch.Tensor(std.size()).normal_())
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
        return rec, mu, z


model = VAE()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_func(input_data, rec, mu, log_var):
    """
    loss function for VAE
    input: x, rec_x, mu, log_var
    return: loss for VAE
    """
    BCE = F.binary_cross_entropy(rec.view(-1, 28*28), input_data.view(-1, 28*28), size_average=False)  #why the size_average is False
    # -0.5*sum(1+log(sigma^2)-mu^2-sigma^2), how this func is constructed
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data)
        data_rec, mu, log_var = model.forward(data)
        loss = loss_func(input_data=data, rec=data_rec, mu=mu, log_var=log_var)

        model.zero_grad()
        loss.backward()
        
        optimizer.step()

        train_loss += loss.data[0]
        if batch_idx % 100 == 0:
            print("==> epoch:{}[{}/{}], training loss:{:.5f}".format(epoch, batch_idx*len(data), len(train_loader.dataset), loss.data[0]))

    print("=> epoch:{}, training loss:{:.5f}".format(epoch, train_loss/len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, (data, label) in enumerate(test_loader):
        data = Variable(data, volatile=True)

        data_rec, mu, log_var = model(data)
        loss = loss_func(data, data_rec, mu, log_var)

        test_loss += loss.data[0]
        
        if batch_idx == 0:
            n = min(8, args.batch_size)
            comparision = torch.cat([data[:n], data_rec.view_as(data)[:n]])
            torchvision.utils.save_image(tensor=comparision.view(-1,1,28,28).data.cpu(), filename=args.root+"./vae_generation/rec/comparision_{}.png".format(epoch), nrow=8)

    print("=> epoch:{}, test loss:{:.5f}".format(epoch, test_loss/len(test_loader.dataset)))



if __name__ == "__main__":
    for epoch in range(args.epoches):
        train(epoch)
        test(epoch)
        #sample
        sampler = torch.rand(64, args.hidden_dim)
        sampler = Variable(sampler, volatile=True)
        gene_sampler = model.decode(sampler)

        torchvision.utils.save_image(tensor=gene_sampler.data.cpu(), filename=args.root+"./vae_generation/rec/gene_sampler_{}.png".format(epoch), nrow=8)
        