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
from tensorboardX import SummaryWriter
from itertools import product

parser = argparse.ArgumentParser(description="plain GAN for mnist")
parser.add_argument("--root", type=str, default="/home/ubuntu/lijiguo/workspace/pytorch_examples", 
                    help="root path for pytorch examples. default /home/lijiguo/workspace/pytorch_examples/")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for training data. default 64")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disable cuda use")
parser.add_argument("--epoches", type=int, default=100, help="training epoches, default 10")
parser.add_argument("--hidden_dim", type=int, default=1024, help="hidden dimension of gan. (default 1024, 3 layers total)")
parser.add_argument("--z_dim", type=int, default=50, help="dimension of z. default 10")
parser.add_argument("--condition_dim", type=int, default=2, help="dimension of c. default 2")
parser.add_argument("--condition_embedding_dim", type=int, default=12, help="embedding dimension of c. default 2")
parser.add_argument("--class_num", type=int, default=10, help="class number for info gan, default 10")
parser.add_argument("--x_feature_dim", type=int, default=512, help="x_feature dim for D, default 512")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("cuda in available, use cuda")
else:
    print("cuda in not available, use cpu")

model_dir = "./info_gan"
log_dir = os.path.join(args.root, model_dir, "./log")
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

writer = SummaryWriter(log_dir=log_dir)

#torch.manual_seed(1)
#if args.cuda:
 #   torch.cuda.manual_seed(1)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=True, download=True, 
                    transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5))
                    ]) ),
    batch_size=args.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=os.path.join(args.root, "../data/mnist"), train=False, download=True,
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
    def __init__(self, noise_dim=10, condition_dim=10, channel=1):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.condition_dim=condition_dim
        self.channel = channel
        self.net = nn.Sequential(
            # batch x 10 x 1 x 1 --> batch x 512 x 3 x 3
            nn.ConvTranspose2d(self.noise_dim+self.condition_dim, 512, 3, 1, 0, bias=False),
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
        self.output = nn.Tanh()
    
    def forward(self, z, c):
        z_c = torch.cat([z,c.unsqueeze(2).unsqueeze(3)], 1)
        x = self.net(z_c)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    """
    discriminator for GAN
    """
    def __init__(self, image_size=28, channel=1, x_feature_dim=512):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.channel = channel
        self.x_feature_dim = x_feature_dim
        self.net = nn.Sequential(
            # 3 x 28 x 28 --> 64 x 24 x 24
            nn.Conv2d(self.channel, self.x_feature_dim//8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(self.x_feature_dim//8),
            nn.LeakyReLU(0.2, True),
            # 64 x 24 x 24 --> 128 x 12 x 12
            nn.Conv2d(self.x_feature_dim//8, self.x_feature_dim//4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.x_feature_dim//4),
            nn.LeakyReLU(0.2, True),
            # 128 x 12 x 12 --> 256 x 6 x 6
            nn.Conv2d(self.x_feature_dim//4, self.x_feature_dim//2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.x_feature_dim//2),
            nn.LeakyReLU(0.2, True),
            # 256 x 6 x 6 --> 512 x 3 x 3
            nn.Conv2d(self.x_feature_dim//2, self.x_feature_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.x_feature_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.image_size_temp = (self.image_size-4)//8
        self.project_layer = nn.Sequential(
            nn.Linear(self.x_feature_dim*self.image_size_temp**2, self.x_feature_dim*3**2, bias=False),
            nn.BatchNorm1d(self.x_feature_dim*3**2),
            nn.LeakyReLU(0.2, True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(self.x_feature_dim, self.x_feature_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.x_feature_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.x_feature_dim, 1, 3, 1, 0),
            nn.Sigmoid()
        )


    def forward(self, z):
        """
        input: z (batch_size x 1 x 28 x 28)
        output: pred (batch_size x 1)
        """
        x_feature_temp = self.net(z).view(-1, self.x_feature_dim*self.image_size_temp**2)
        x_feature = self.project_layer(x_feature_temp).view(-1,self.x_feature_dim, 3, 3)
        x = self.output(x_feature)
        return x.squeeze(), x_feature


class Q_net(nn.Module):
    def __init__(self, feature_shape, class_num, condition_dim):
        """
        x_feature: batch_size x feature_shape
        """
        super(Q_net, self).__init__()
        self.feature_shape = feature_shape
        self.class_num = class_num
        self.condition_dim = condition_dim

        self.classifier = nn.Sequential(
            nn.Conv2d(self.feature_shape[0], self.feature_shape[0], 3,1,1, bias=False),
            nn.BatchNorm2d(self.feature_shape[0]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.feature_shape[0], self.class_num, self.feature_shape[2],1,0, bias=True)
        )

        self.distribution = nn.Sequential(
            nn.Conv2d(self.feature_shape[0], self.feature_shape[0], 3,1,1, bias=False),
            nn.BatchNorm2d(self.feature_shape[0]),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.feature_shape[0], self.condition_dim*2, self.feature_shape[2],1,0, bias=True)
        )

    def forward(self, x):
        """
        Input:
            x: batch_size x feature_shape tensor

        Return:
            pred_class:
            mu:
            log_var:
        """
        pred_class = self.classifier(x).squeeze(3).squeeze(2)
        temp = self.distribution(x)
        mu, log_var = temp[:,:self.condition_dim], temp[:,self.condition_dim:]
        mu, log_var = mu.squeeze(3).squeeze(2), log_var.squeeze(3).squeeze(2)
        return pred_class, mu, log_var


def loss_func_D(input, target):
    error = F.binary_cross_entropy(input, target)
    return error

def loss_func_Q(input, target):
    error = F.cross_entropy(input, target)
    return error 

def loss_func_guanssian(x, mu, log_var):
    """get the PDF of N(mu, log_var**2) as x
    reference: https://github.com/pianomania/infoGAN-pytorch/blob/master/trainer.py
    """
    # logli = -0.5*torch.log(log_var.pow(2).mul(2*np.pi)+1e-6) - \
    #         (x-mu).pow(2).div(log_var.pow(2).mul(2)+1e-6)
    # logli = -0.5*torch.log(torch.Tensor([2*np.pi])) - \
    #         0.5*(x-mu).pow(2).div(1)
    return 0.5*F.mse_loss(mu, x)
    # return 1 - logli.mean(dim=1).mean()
    # logli = F.binary_cross_entropy_with_logits(mu, x)
    # return logli 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

model_G = Generator(noise_dim=args.z_dim, condition_dim=args.condition_dim+args.class_num)
model_D = Discriminator()
model_Q = Q_net((args.x_feature_dim, 3,3), args.class_num, args.condition_dim)

model_G, model_D, model_Q = model_G.apply(weights_init), model_D.apply(weights_init), model_Q.apply(weights_init)
if args.cuda:
    model_G.cuda()
    model_D.cuda()
    model_Q.cuda()

optimizer_G = torch.optim.Adam(model_G.parameters(), lr=1e-3, betas=(0.5, 0.999))# torch.optim.SGD(model_G.parameters(), lr=1e-3, momentum=0.9)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=2e-4, betas=(0.5, 0.999))# torch.optim.SGD(model_D.parameters(), lr=1e-4, momentum=0.9)
optimizer_Q = torch.optim.Adam(model_Q.parameters(), lr=2e-4, betas=(0.5, 0.999))

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
    for (data, label) in loader_bar:
        batch_cnt += 1
        batch_size = data.size(0)
        #real data
        data_real = data
        label_real = torch.Tensor(batch_size).requires_grad_(False).fill_(1)
        data_real = Variable(data_real)
        condition_class_idx = torch.randint(args.class_num, size=(batch_size,)).long()  # target for Q
        condition_class = torch.eye(args.class_num)[condition_class_idx.long()].reshape(batch_size, args.class_num)
        # condition_class[range(batch_size), condition_class_idx] = 1.0
        condition_other = (torch.rand(size=(batch_size, args.condition_dim))-0.5)/0.5
        #condition_other = torch.rand(size=(batch_size, args.condition_dim))
        condition = torch.cat([condition_class, condition_other], 1).requires_grad_(True)
        #condition = torch.eye(10)[label].reshape(batch_size, 10)

        if args.cuda:
            data_real, label_real, condition, condition_class_idx, condition_other = \
                data_real.cuda(), label_real.cuda(), condition.cuda(), condition_class_idx.cuda(), condition_other.cuda()

        pred_real, _ = model_D.forward(data_real)
        
        #fake data
        z = torch.randn(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_(True)
        label_fake = torch.Tensor(batch_size).requires_grad_(False).fill_(0)
        if args.cuda:
            z = z.cuda()
            label_fake = label_fake.cuda()

        data_fake = model_G.forward(z, condition)

        #update D
        model_D.zero_grad()
        model_Q.zero_grad()
        pred_D_real, _ = model_D.forward(data_real)
        pred_D_fake, x_feature_fake = model_D.forward(data_fake.detach())
        pred_class, mu, log_var = model_Q.forward(x_feature_fake)
        loss_D_real = loss_func_D(pred_D_real, label_real)
        loss_D_fake = loss_func_D(pred_D_fake, label_fake)
        loss_D_gan = (loss_D_real+loss_D_fake)/2
        loss_D_Q = loss_func_Q(pred_class, condition_class_idx)
        loss_D_guassian = loss_func_guanssian(condition_other, mu, log_var)

        loss_D = loss_D_gan + loss_D_Q + loss_D_guassian
        loss_D.backward()
        optimizer_Q.step()
        optimizer_D.step()

        #update G
        z = torch.randn(size=(batch_size, args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_()
        if args.cuda:
            z = z.cuda()
        data_fake = model_G.forward(z, condition)
        pred_G, x_feature_G = model_D.forward(data_fake)
        pred_class, mu, log_var = model_Q.forward(x_feature_G)

        model_G.zero_grad()
        model_Q.zero_grad()
        loss_G_gan = loss_func_D(pred_G, label_real)
        loss_G_Q = loss_func_Q(pred_class, condition_class_idx.long())
        loss_G_guassian = loss_func_guanssian(condition_other, mu, log_var)
        loss_G = loss_G_gan + loss_G_Q + loss_G_guassian
        loss_G.backward()
        optimizer_G.step() 
        optimizer_Q.step()
        loader_bar.set_description("==>epoch:{:2d}, loss_D:{:6.4f}[{:6.4f},{:6.4f},{:6.4f},{:6.4f}], loss_G:{:6.4f}[{:6.4f},{:6.4f},{:6.4f}]".format(
                    epoch, 
                    loss_D.cpu().item(), loss_D_real.cpu().item(), loss_D_fake.cpu().item(), loss_D_Q.cpu().item(), loss_D_guassian.cpu().item(),
                    loss_G.cpu().item(), loss_G_gan.cpu().item(), loss_G_Q.cpu().item(), loss_G_guassian.cpu().item(),
                    ))
        writer.add_scalars(main_tag="loss_d", tag_scalar_dict={
            "loss_d_real":loss_D_real.cpu().item(),
            "loss_d_fake":loss_D_fake.cpu().item(),
            "loss_d_q":loss_D_Q.cpu().item(),
            "loss_d_guassian":loss_D_guassian.cpu().item()
        }, global_step=batch_cnt)
        writer.add_scalars(main_tag="loss_g", tag_scalar_dict={
            "loss_g_gan":loss_G_gan.cpu().item(),
            "loss_g_q":loss_G_Q.cpu().item(),
            "loss_g_guassian":loss_G_guassian.cpu().item()
        }, global_step=batch_cnt)
        writer.add_scalars(main_tag="loss", tag_scalar_dict={
            "loss_d":loss_D.cpu().item(),
            "loss_g":loss_G.cpu().item()
        }, global_step=batch_cnt)
        if batch_cnt%100 == 0:
            #save image
            if not os.path.exists(os.path.join(args.root, model_dir, "./sample/")):
                os.system("mkdir {}".format(os.path.join(args.root, model_dir, "./sample/")))
            # sample
            sample_data1, sample_data2 = sample(model_G)
            torchvision.utils.save_image(tensor=sample_data1.view(-1,1,28,28).data.cpu(), 
                    filename=os.path.join(args.root, model_dir, "./sample/", "sample_fake1_{}.png".format(save_idx)), nrow=10)
            torchvision.utils.save_image(tensor=sample_data2.view(-1,1,28,28).data.cpu(), 
                    filename=os.path.join(args.root, model_dir, "./sample/", "sample_fake1_{}.png".format(save_idx)), nrow=10)
            writer.add_image(tag="fake_image1", img_tensor=torchvision.utils.make_grid(sample_data1, normalize=True, scale_each=True, nrow=10, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="fake_image2", img_tensor=torchvision.utils.make_grid(sample_data2, normalize=True, scale_each=True, nrow=10, range=(-1,1)), global_step=batch_cnt)
            writer.add_image(tag="real_image", img_tensor=torchvision.utils.make_grid(data_real, normalize=True, scale_each=True, range=(-1,1)), global_step=batch_cnt)
            #save model
            if not os.path.exists(os.path.join(args.root, model_dir, "./model/")):
                os.system("mkdir {}".format(os.path.join(args.root, model_dir, "./model/")))
            
            #torch.save(model_G.state_dict(), f=os.path.join(args.root, model_dir, "./model/model_G_{}.pytorch".format(save_idx)))
            #torch.save(model_D.state_dict(), f=os.path.join(args.root, model_dir, "./model/model_D_{}.pytorch".format(save_idx)))
            save_idx += 1

def sample(net_G):
    
    condition_class_idx = np.array(list(range(args.class_num)))
    condition_class = np.eye(args.class_num)[condition_class_idx.astype(np.int32)]

    condition_other1 = np.stack([np.linspace(-1,1,10), np.zeros(10)]).transpose()
    condition_other2 = np.stack([np.zeros(10), np.linspace(-1,1,10)]).transpose()
    # condition_other1 = np.stack([np.linspace(0,1,10), np.zeros(10)]).transpose()
    # condition_other2 = np.stack([np.zeros(10), np.linspace(0,1,10)]).transpose()
    condition_np1 = [np.concatenate([_class, _other]) for _class in condition_class for _other in condition_other1]
    condition_np2 = [np.concatenate([_class, _other]) for _class in condition_class for _other in condition_other2]
    #condition_np = list(product(condition_class, condition_other))

    condition1 = torch.Tensor(condition_np1)
    condition2 = torch.Tensor(condition_np2)

    #condition = torch.Tensor([torch.cat([_class, _other], 0) for _class in condition_class for _other in condition_other]).requires_grad_(False)
    z = torch.randn(size=(condition1.shape[0], args.z_dim)).unsqueeze(2).unsqueeze(3).requires_grad_(False)
    if args.cuda:
        condition1, condition2, z = condition1.cuda(), condition2.cuda(), z.cuda()
    data_fake1, data_fake2 = net_G.forward(z, condition1), net_G.forward(z, condition2)
    return data_fake1, data_fake2

if __name__=="__main__":
    for epoch in range(0, args.epoches):
        train(epoch)


        