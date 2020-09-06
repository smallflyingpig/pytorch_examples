import argparse
import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
import torchvision
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description="VAE mnist examples")
parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training.(default 64)")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda training")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate, default 1e-4")
parser.add_argument("--epoches", type=int, default=100, help="set the training epoches. default(10)")
parser.add_argument("--hidden_dim", type=int, default=20, help="dimension of encoding data.(default 20)")
parser.add_argument("--root", type=str, default="/home/jiguo/pytorch_examples", 
                    help="root path for 'pytorch_examples', default '/home/jiguo/pytorch_examples'")
parser.add_argument("--model", type=str, default="mlp", 
                    help="model type for VAE encoder and decoder, mlp or conv, decault mlp")
parser.add_argument("--img_size", type=int, default=32, 
                    help="image size the image will resize to, default 32")                    
parser.add_argument("--KLD_coeff", type=float, default=3, 
                    help="loss coeff of KLD, default 1")    
parser.add_argument("--img_channel", type=int, default=3, 
                    help="image channel, default 3")                                    

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("cuda in available, use cuda")
else:
    print("cuda in not available, use cpu")


model_dir = "vae_generation"
rec_dir = os.path.join(args.root, model_dir, "./rec")
if not os.path.exists(rec_dir):
    os.mkdir(rec_dir)

log_dir = os.path.join(args.root, model_dir, "./log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

writer = SummaryWriter(log_dir)


torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)

param = {"num_workers":4, "pin_memory":True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=args.root + "./data/cifar10", train=True, download=False, 
                    transform=transforms.Compose([
                    transforms.Resize(size=(args.img_size)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ])),
    batch_size=args.batch_size, shuffle=True, **param
)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root=args.root + "./data/cifar10", train=False, 
                    transform=transforms.Compose([
                    transforms.Resize(size=(args.img_size)),
                    transforms.ToTensor(),
                    #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
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
            ResBlock(self.ref_dim, activate="LeakyReLU"),
            # img_size//2 --> img_size//4
            nn.Conv2d(self.ref_dim, self.ref_dim*2, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*2),
            nn.LeakyReLU(0.2, True),
            ResBlock(self.ref_dim*2, activate="LeakyReLU"),
            # img_size//4 --> img_size//8
            nn.Conv2d(self.ref_dim*2, self.ref_dim*4, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*4),
            nn.LeakyReLU(0.2, True),
            ResBlock(self.ref_dim*4, activate="LeakyReLU")
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
            ResBlock(self.ref_dim*4, activate="ReLU"),
            # ref_dim*4 x img_size//8 x img_size//8 --> ref_dim*2 x img_size//4 x img_size//4
            nn.ConvTranspose2d(self.ref_dim*4, self.ref_dim*2, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*2),
            nn.ReLU(True),
            ResBlock(self.ref_dim*2, activate="ReLU"),
            # ref_dim*2 x img_size//4 x img_size//4 --> ref_dim*1 x img_size//2 x img_size//2
            nn.ConvTranspose2d(self.ref_dim*2, self.ref_dim*1, 4,2,1, bias=False),
            nn.BatchNorm2d(self.ref_dim*1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.ref_dim, self.img_channel, 4,2,1, bias=True),
        )
        self.output = nn.Sequential(
            nn.Conv2d(self.img_channel, self.img_channel, 3,1,1, bias=False),
        )
        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.img_channel, self.ref_dim, 3,1,1, bias=False),
            nn.BatchNorm2d(self.ref_dim),
            nn.ReLU(True),
            ResBlock(self.ref_dim),
            nn.Conv2d(self.ref_dim, self.img_channel, 3,1,1, bias=False),
        )
        self.output1 = nn.Sequential(
            nn.Conv2d(self.img_channel, self.img_channel, 3,1,1, bias=False),
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
        out0 = self.output(x)
        out0_sigmoid = F.sigmoid(out0)
        x = self.decoder1(x-out0)
        out1 = self.output1(x)
        out1_sigmoid = F.sigmoid(out1)
        return out0_sigmoid, out1_sigmoid

    def forward(self, input_data):
        mu, log_var = self.encode(input_data)
        z = self.reparameter(mu, log_var)
        rec = self.decode(z)
        return rec, mu, log_var

def quantization(image, bit=int(8)):
    """
    image: batch x plane x W x H
    """
    batch, plane, width, height = image.shape
    image_out = np.zeros(shape=(batch, plane*bit, width, height))
    image_byte = (image.cpu().numpy()*(2**bit-1)).astype(np.uint8)
    for idx in range(int(plane*bit)):
        plane_idx = idx//bit
        bit_idx = idx%bit
        bit_mask = np.array([int(1)<<(bit-bit_idx-1)], dtype=np.uint8)
        image_out[:,idx,:,:] = (np.bitwise_and(image_byte[:,plane_idx,:,:], bit_mask)>0)

    return torch.Tensor(image_out).type_as(image)


def dequantization(image, plane=3, bit_max=int(8), norm_flag=False):
    """
    image: batch x (plane*bit) x W x H
    """
    batch, plane_bit, width, height = image.shape
    bit = plane_bit//plane
    image_out = torch.zeros(size=(batch, plane, width, height)).type_as(image)
    for idx in range(int(bit)):
        for plane_idx in range(plane):
            if norm_flag:
                image_out[:,plane_idx,:,:] += (image[:,int(idx+bit*plane_idx),:,:]>0.5).type_as(image)* 2**(bit_max-idx-1)
            else:
                image_out[:,plane_idx,:,:] += (image[:,int(idx+bit*plane_idx),:,:])* 2**(bit_max-idx-1)
            
    return (image_out/(2**bit_max-1))


def cat_quan_data(image_seq, plane=3):
    batch_size, _, width, height = image_seq[0].shape
    bit = 0
    for image in image_seq:
        bit += image.shape[1]//plane
    image_out = torch.zeros(batch_size, int(bit*plane), width, height)
    
    start_idx = 0
    for image in image_seq:
        bit_num = image.shape[1]//plane
        for plane_idx in range(plane):
            image_out[:,start_idx+plane_idx*bit:start_idx+bit_num+plane_idx*bit,:,:] = image[:,plane_idx*bit_num:plane_idx*bit_num+bit_num,:,:]
        start_idx += bit_num
    return image_out


def loss_func_quan(image, target, bit=int(8)):
    batch_size, plane_bit, width, height = image.shape
    plane_num = plane_bit//bit
    weight = torch.ones_like(image).type_as(image)
    for bit_idx in range(bit):
        for plane_idx in range(plane_num):
            weight[:,int(bit_idx+plane_idx*bit),:,:] = 2**(bit-bit_idx-1)

    loss = F.binary_cross_entropy(image, target, weight=weight, reduce=False).sum(3).sum(2).mean()
    return loss
    
def loss_func_BCE(image, target):
    batch_size = image.shape[0]
    #BCE = F.mse_loss(image.view(batch_size, -1), target.view(batch_size, -1), reduce=False).sum(dim=1).mean()  #why the size_average is False
    BCE = F.binary_cross_entropy(image.view(batch_size, -1), target.view(batch_size, -1), reduce=False).sum(dim=1).mean()  #why the size_average is False
    return BCE


def loss_func_KLD(mu, log_var):
    """
    loss function for VAE
    input: x, rec_x, mu, log_var
    return: loss for VAE
    """
    batch_size = mu.shape[0]
    # BCE = F.mse_loss(rec.view(batch_size, -1), input_data.view(batch_size, -1), reduce=False).sum(dim=1).mean()  #why the size_average is False
    # -0.5*sum(1+log(sigma^2)-mu^2-sigma^2), how this func is constructed
    # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1))
    return KLD


#model = VAE_conv(img_size=(args.img_size, args.img_size), img_channel=args.img_channel)
model = VAE_conv(img_size=(args.img_size, args.img_size), img_channel=args.img_channel)

if args.cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


batch_cnt = 0
def train(epoch):
    global batch_cnt
    model.train()
    train_loss = 0
    loader_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        batch_cnt += 1
        data_quan = quantization(data.requires_grad_(False))
        label_quan, label_quan1 = torch.zeros_like(data), torch.zeros_like(data)
        label_quan[:,0,:,:], label_quan[:,1,:,:], label_quan[:,2,:,:] = \
            data_quan[:,0,:,:], data_quan[:,8,:,:], data_quan[:,16,:,:]
        label_quan1[:,0,:,:], label_quan1[:,1,:,:], label_quan1[:,2,:,:] = \
            data_quan[:,1,:,:], data_quan[:,9,:,:], data_quan[:,17,:,:]
        data_temp = dequantization(data_quan)
        data = data.requires_grad_(True)
        if args.cuda:
            data, label_quan, label_quan1 = data.cuda(), label_quan.cuda(), label_quan1.cuda()

        (data_rec, data_rec1), mu, log_var = model.forward(data)
        #loss_BCE = loss_func_BCE(data_rec, data.detach().requires_grad_(False))
        loss_BCE = loss_func_BCE(data_rec, label_quan.detach().requires_grad_(False))
        loss_BCE1 = loss_func_BCE(data_rec1, label_quan1.detach().requires_grad_(False))
        #loss_BCE = loss_func_quan(data_rec, data.detach().requires_grad_(False))
        loss_KLD = loss_func_KLD(mu=mu, log_var=log_var)
        loss = loss_BCE + args.KLD_coeff*loss_KLD

        model.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().item()
        loader_bar.set_description("[{:5s}] epoch:{:5d}, loss:{:10.5f}[{:10.5f},{:10.5f},{:10.5f}]".format("train", epoch, 
            loss.cpu().item(), loss_BCE.cpu().item(), loss_BCE1.cpu().item(), loss_KLD.cpu().item()
            ))
        writer.add_scalars(main_tag="loss", tag_scalar_dict={
            "loss_BCE":loss_BCE.cpu().item(),
            "loss_KLD":loss_KLD.cpu().item()
        }, global_step=batch_cnt)
        
        if batch_cnt %100 == 0:
            data_rec_save, data_save = \
                dequantization(cat_quan_data([data_rec]), norm_flag=True), dequantization(cat_quan_data([label_quan]), norm_flag=True)
            writer.add_image(tag="image_fake", img_tensor=torchvision.utils.make_grid(data_rec_save, normalize=True, range=(0,1)))
            writer.add_image(tag="image_real", img_tensor=torchvision.utils.make_grid(data_save, normalize=True, range=(0,1)))
            
    print("[{:5s}] epoch:{:5d}, loss:{:10.5f}".format("train", epoch, train_loss/(len(train_loader))))



def test(epoch):
    model.eval()
    test_loss = 0
    loader_bar = tqdm.tqdm(test_loader)
    for batch_idx, (data, label) in enumerate(loader_bar):
        data = data.requires_grad_(False)
        data = quantization(data)
        if args.cuda:
            data = data.cuda()

        (data_rec, data_rec1), mu, log_var = model(data)
        loss_BCE = loss_func_quan(data_rec, data.detach().requires_grad_(False))
        loss_KLD = loss_func_KLD(mu, log_var)
        loss = loss_BCE + args.KLD_coeff*loss_KLD

        test_loss += loss.cpu().item()
        
        loader_bar.set_description("[{:5s}] epoch:{:5d}, loss:{:10.5f}[{:10.5f},{:10.5f}]".format("test", epoch, 
            loss.cpu().item(), loss_BCE.cpu().item(), loss_KLD.cpu().item()
            ))

        if batch_idx == 0:
            n = min(8, args.batch_size)
            data_save, data_rec_save = dequantization(data), dequantization(data_rec)
            comparision = torch.cat([data_save[:n], data_rec_save.view_as(data_save)[:n]])
            if not os.path.exists(rec_dir):
                os.system("mkdir {}".format(rec_dir))

            torchvision.utils.save_image(tensor=comparision.view(-1,args.img_channel, args.img_size,args.img_size).data.cpu(), 
                    filename=os.path.join(rec_dir, "./comparision_{}.png".format(epoch)), normalize=True, range=(-1,1), nrow=8)

    print("[{:5s}] epoch:{:5d}, loss:{:10.5f}".format("test", epoch, test_loss/len(test_loader)))



if __name__ == "__main__":
    for epoch in range(args.epoches):
        train(epoch)
        #test(epoch)
        #sample
        model.eval()
        sampler = torch.randn(64, args.hidden_dim)
        sampler = sampler.requires_grad_(False)
        if args.cuda:
            sampler = sampler.cuda()
        gene_sampler = model.decode(sampler)
        gene_sampler = dequantization(cat_quan_data([gene_sampler[0]]))
        #gene_sampler = dequantization(gene_sampler)
        
        if not os.path.exists(rec_dir):
                os.system("mkdir {}".format(rec_dir))
                
        torchvision.utils.save_image(tensor=gene_sampler.cpu(), 
                filename=os.path.join(rec_dir, "./gene_sampler_{}.png".format(epoch)), normalize=True, range=(0,1), nrow=8)
        