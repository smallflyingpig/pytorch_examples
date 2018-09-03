import argparse
import os
import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision
import torchvision.utils as vutils
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import time


parser = argparse.ArgumentParser(description="VAE mnist examples")
parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training.(default 64)")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda training")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate, default 1e-4")
parser.add_argument("--epoches", type=int, default=100, help="set the training epoches. default(10)")
parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of encoding data.(default 20)")
parser.add_argument("--noise_dim", type=int, default=10, help="noise dim of generator.(default 10)")
parser.add_argument("--root", type=str, default="/home/jiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")
parser.add_argument("--model", type=str, default="mlp", 
                    help="model type for VAE encoder and decoder, mlp or conv, decault mlp")
parser.add_argument("--img_size", type=int, default=32, 
                    help="image size the image will resize to, default 32")                    
parser.add_argument("--KLD_coeff", type=float, default=5, 
                    help="loss coeff of KLD, default 5")   
parser.add_argument("--img_channel", type=int, default=3, 
                    help="image channel for data set, default 3 for cifar")                                     

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

model_dir = "vae_generation"
rec_dir = os.path.join(args.root, model_dir, "./rec")
if not os.path.exists(rec_dir):
    os.mkdir(rec_dir)

log_dir = os.path.join(args.root, model_dir, "./log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

time_now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
log_dir_cur = os.path.join(log_dir, time_now)
os.mkdir(log_dir_cur)

writer = SummaryWriter(log_dir_cur)

if args.cuda:
    print("cuda in available, use cuda")
else:
    print("cuda in not available, use cpu")

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)

param = {"num_workers":4, "pin_memory":True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=args.root + "./data/cifar10", train=True, download=False, 
                    transform=transforms.Compose([
                    transforms.Resize(size=(args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ])),
    batch_size=args.batch_size, shuffle=True, **param
)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=args.root + "./data/cifar10", train=False, 
                    transform=transforms.Compose([
                    transforms.Resize(size=(args.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ])),
    batch_size=args.batch_size, shuffle=True, **param
)


class ResBlock(nn.Module):
    def __init__(self, channel, activate="ReLU"):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.block = nn.Sequential(
            nn.Conv2d(self.channel, self.channel//4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel//4),
            nn.ReLU(True),
            nn.Conv2d(self.channel//4, self.channel//4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel//4),
            nn.ReLU(True),
            nn.Conv2d(self.channel//4, self.channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.channel),
        )
        if activate=="ReLU":
            self.activate = nn.ReLU(True)
        else:
            self.activate = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        residual = self.block(x)
        return self.activate(x+residual)

class VAE_conv(nn.Module):
    def __init__(self, img_size, img_channel=1):
        super(VAE_conv, self).__init__()
        self.hidden_dim = args.hidden_dim
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
            nn.ConvTranspose2d(self.hidden_dim, self.ref_dim*4, self.img_size[0]//8,1,0, bias=False),
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
            nn.Tanh()
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

    def decode(self, z):
        x = self.decoder(z.unsqueeze(2).unsqueeze(3))
        return x.view(-1, self.img_channel, self.img_size[0], self.img_size[1])

    def forward(self, input_data):
        mu, log_var = self.encode(input_data)
        z = self.reparameter(mu, log_var)
        rec = self.decode(z)
        return rec, mu, log_var

class Generator(nn.Module):
    """
    generator for GAN
    """
    def __init__(self, noise_dim=10, condition_dim=1, condition_size=(28,28), img_channel=1):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.condition_size = condition_size
        self.img_channel = img_channel
        self.ref_dim = 128

        self.G = nn.Sequential(
            # condition_size//4
            nn.ConvTranspose2d(self.noise_dim, self.ref_dim*4, self.condition_size[0]//4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ref_dim*4),
            nn.ReLU(True),
            # condition_size//2
            nn.ConvTranspose2d(self.ref_dim*4, self.ref_dim*2, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*2),
            nn.ReLU(True),
            # condition_size//1
            nn.ConvTranspose2d(self.ref_dim*2, self.ref_dim, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim),
            nn.ReLU(True)
        )
        self.net = nn.Sequential(
            nn.Conv2d(self.ref_dim+self.condition_dim*self.img_channel, self.ref_dim, 3,1,1, bias=False),
            nn.BatchNorm2d(self.ref_dim),
            nn.ReLU(True),
            ResBlock(self.ref_dim, activate="ReLU"),
            nn.Conv2d(self.ref_dim, self.img_channel, 3,1,1, bias=True),
        )
        self.output = nn.Tanh()
    
    def forward(self, z, c):
        out_G = self.G(z.unsqueeze(2).unsqueeze(3))

        out_G_c = torch.cat([out_G,c], 1)
        x = self.net(out_G_c)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self, img_size, condition_dim=1, img_channel=1):
        super(Discriminator, self).__init__()
        self.img_size = img_size 
        self.condition_dim = condition_dim
        self.img_channel = img_channel
        self.ref_dim = 32
        self.net = nn.Sequential(
            # img_size --> img_size//2
            nn.Conv2d(self.img_channel+self.condition_dim*self.img_channel, self.ref_dim, 4,2,1, bias=False),
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
            ResBlock(self.ref_dim*4, activate="LeakyReLU")
        )
        self.output = nn.Sequential(
            nn.Conv2d(self.ref_dim*4, 1, self.img_size[0]//8, 1, 0),
            nn.Sigmoid()
        )


    def forward(self, z, c):
        """
        input: z (batch_size x 1 x img_size[0] x img_size[1])
        output: pred (batch_size x 1)
        """
        z_c = torch.cat([z,c], 1)
        x_feature = self.net(z_c)
        x = self.output(x_feature)
        return x.squeeze()


model_VAE = VAE_conv(img_size=(args.img_size, args.img_size), img_channel=args.img_channel)

model_D = Discriminator(img_size=(args.img_size, args.img_size), img_channel=args.img_channel)
model_G = Generator(noise_dim=args.noise_dim, condition_size=(args.img_size, args.img_size), img_channel=args.img_channel)
if args.cuda:
    model_VAE.cuda()
    model_D.cuda()
    model_G.cuda()

optimizer_VAE = torch.optim.Adam(model_VAE.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr)
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.lr)

def loss_func_BCE(input, target):
    error = F.binary_cross_entropy(input, target)
    return error


def loss_func(data_rec, data):
    batch_size, channel = data.shape[0], data.shape[1]
    error = F.mse_loss(data_rec.view(batch_size, -1), data.view(batch_size, -1), reduce=False).sum()/(batch_size*channel)
    return error

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

batch_cnt = 0
def train(epoch):
    global batch_cnt
    model_VAE.train()
    model_D.train()
    model_G.train()
    train_loss_D = 0
    train_loss_VAE = 0
    train_loss_G = 0
    loader_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        batch_cnt += 1
        batch_size = data.shape[0]
        data = data.requires_grad_(True)
        label_real = torch.ones(batch_size)
        label_fake = torch.zeros(batch_size)
        noise_G = torch.randn(size=(batch_size, args.noise_dim))
        if args.cuda:
            data, label_real, label_fake = data.cuda(), label_real.cuda(), label_fake.cuda()
            noise_G = noise_G.cuda()

        # update D
        data_rec, mu, log_var = model_VAE.forward(data)
        condition = data_rec.detach()
        data_fake = model_G.forward(noise_G, condition)
        pred_real, pred_fake = model_D(data, condition), model_D(data_fake.detach(), condition)
        loss_D_real, loss_D_fake = loss_func_BCE(pred_real, label_real), loss_func_BCE(pred_fake, label_fake)
        loss_D = loss_D_real + loss_D_fake

        model_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        #update VAE
        model_VAE.zero_grad()
        loss_VAE_KLD = loss_func_KLD(mu=mu, log_var=log_var)
        loss_VAE_BCE = loss_func(data_rec, data.detach().requires_grad_(False))
        loss_VAE = loss_VAE_BCE + args.KLD_coeff*loss_VAE_KLD
        model_VAE.zero_grad()
        loss_VAE.backward()
        optimizer_VAE.step()

        #update G
        noise_G = torch.randn(size=(batch_size, args.noise_dim))
        if args.cuda:
            noise_G = noise_G.cuda()
        data_fake = model_G.forward(noise_G, condition)
        pred_fake = model_D(data_fake, condition)
        loss_G_fake = loss_func_BCE(pred_fake, label_real)
        loss_G = loss_G_fake
        model_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()


        train_loss_D += loss_D.cpu().item()
        train_loss_VAE += loss_VAE.cpu().item()
        train_loss_G += loss_G.cpu().item()

        loader_bar.set_description("[{:5s}] epoch:{:5d}, loss D:{:6.4f}, loss VAE:{:6.4}, loss G:{:6.4}".format("train", epoch, 
            loss_D.cpu().item(), loss_VAE.cpu().item(), loss_G.cpu().item()
            ))
        writer.add_scalars(main_tag="loss", tag_scalar_dict={
            "loss_d":loss_D.cpu().item(),
            "loss_vae":loss_VAE.cpu().item(),
            "loss_g":loss_G.cpu().item()
        }, global_step=batch_cnt)
        writer.add_scalars(main_tag="loss_d", tag_scalar_dict={
            "loss_d_real":loss_D_real.cpu().item(),
            "loss_d_fake":loss_D_fake.cpu().item()
        }, global_step=batch_cnt)
        writer.add_scalars(main_tag="loss_vae", tag_scalar_dict={
            "loss_BCE":loss_VAE_BCE.cpu().item(),
            "loss_KLD":loss_VAE_KLD.cpu().item()
        }, global_step=batch_cnt)
        writer.add_scalars(main_tag="loss_g", tag_scalar_dict={
            "loss_g_fake":loss_G_fake.cpu().item()
        }, global_step=batch_cnt)
        if batch_idx == 0:
            writer.add_image(tag="image_rec", img_tensor=vutils.make_grid(data_rec, normalize=True, range=(-1,1)))
            writer.add_image(tag="image_fake", img_tensor=vutils.make_grid(data_fake, normalize=True, range=(-1,1)))
            writer.add_image(tag="image_real", img_tensor=vutils.make_grid(data, normalize=True, range=(-1,1)))


        
            
    print("[{:5s}] epoch:{:5d}, loss D:{:6.4f}, loss VAE:{:6.4f}".format(
        "train", epoch, train_loss_D/(len(train_loader)), train_loss_VAE/(len(train_loader))
        ))





def sample(epoch):
    model_VAE.eval()
    #sample
    sampler = torch.randn(64, args.hidden_dim)
    sampler = sampler.requires_grad_(False)
    noise = torch.randn(64, args.noise_dim)
    if args.cuda:
        sampler, noise = sampler.cuda(), noise.cuda()
    gene_sampler = model_VAE.decode(sampler)
    
    gene_sampler = model_G.forward(noise, gene_sampler)
    
    if not os.path.exists(rec_dir):
            os.system("mkdir {}".format(rec_dir))
            
    torchvision.utils.save_image(tensor=gene_sampler.data.cpu(), 
            filename=os.path.join(rec_dir, "./gene_sampler_{}.png".format(epoch)), normalize=True, range=(-1,1), nrow=8)
        


if __name__ == "__main__":
    for epoch in range(args.epoches):
        train(epoch)
        sample(epoch)
        