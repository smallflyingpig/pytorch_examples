import sys
import os
import argparse
import logging
import tqdm
sys.path.append(os.getcwd())

import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import grad
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import wgan_gp.utils_celebA as utils


parser = argparse.ArgumentParser("wgan gp celebA")
parser.add_argument("--data_root", type=str, default="/media/ubuntu/Elements/dataset/CelebA/Img/img_align_celeba_all/", help="")
parser.add_argument("--image_size", type=int, default=64, help="output image size, 64, 128 or 256")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu id for training")
parser.add_argument("--epochs", type=int, default=100, help="total epoch for training")
parser.add_argument("--batch_size", type=int, default=64, help="batch size for model training")
parser.add_argument("--n_critic", type=int, default=5, help="iterations for training D when training G once")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate for training model")
parser.add_argument("--model_path", type=str, default="./wgan_gp/model", help="path for checkpoint")
parser.add_argument("--log_dir", type=str, default="./wgan_gp/log", help="log dir")
parser.add_argument("--sample_dir", type=str, default="./wgan_gp/sample", help="path for sample")
parser.add_argument("--num_worker", type=int, default=4, help="number workers for dataloader")
parser.add_argument("--z_dim", type=int, default=100, help="dim for noise")
parser.add_argument("--train", action='store_true', default=False, help="training (not eval or test)")
parser.add_argument("--sample_output_path", type=str, default="./wgan_gp_sample", help="sample output path(test mode)")
parser.add_argument("--sample_num", type=int, default=64, help="total sample image number")

args, _ = parser.parse_known_args()

""" gpu """
utils.cuda_devices([args.gpu_id])

""" run """
writer = tensorboardX.SummaryWriter(args.log_dir)

if args.image_size == 64:
    import wgan_gp.models_64x64 as models
elif args.image_size == 128:
    import wgan_gp.models_128x128 as models
elif args.image_size == 256:
    raise NotImplementedError
    # import wgan_gp.models_25x256 as models
else:
    raise NotImplementedError

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = utils.cuda(torch.rand(shape).requires_grad_(True))
    z = x + alpha * (y - x)

    # gradient penalty
    z = utils.cuda(z)
    o = f(z)
    g = grad(o, z, grad_outputs=utils.cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp


def get_transform_celebA():
    """ data """
    crop_size = 108
    re_size = args.image_size
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(crop),
         transforms.ToPILImage(),
         transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    return transform


def train(G, D, g_optimizer, d_optimizer, data_loader, ckpt_dir, epochs, start_epoch=0):
    z_sample = torch.randn(100, args.z_dim).requires_grad_(True)
    z_sample = utils.cuda(z_sample)
    for epoch in range(start_epoch, epochs):
        for i, (imgs, _) in enumerate(data_loader):
            # step
            step = epoch * len(data_loader) + i + 1
            imgs = imgs.requires_grad_(True)
            # set train
            G.train()
    
            # leafs
            bs = imgs.size(0)
            z = torch.randn(bs, args.z_dim).requires_grad_(True)
            imgs, z = utils.cuda([imgs, z])
    
            f_imgs = G(z)
    
            # train D
            r_logit = D(imgs)
            f_logit = D(f_imgs.detach())
    
            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = gradient_penalty(imgs.data, f_imgs.data, D)
            d_loss = -wd + gp * 10.0
    
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()
    
            writer.add_scalars('loss',
                {
                    'g_wd': wd.data.cpu().numpy(),
                    'g_gp': gp.data.cpu().numpy()
                }, global_step=step
            )
    
            if step % args.n_critic == 0:
                # train G
                z = utils.cuda(torch.randn(bs, args.z_dim).requires_grad_(True))
                f_imgs = G(z)
                f_logit = D(f_imgs)
                g_loss = -f_logit.mean()
    
                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                g_optimizer.step()
    
                writer.add_scalars('loss',
                                   {"g_loss": g_loss.data.cpu().numpy()},
                                   global_step=step)
    
            if (i + 1) % 100 == 0:
                print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, len(data_loader)))
    
            if (i + 1) % 5000 == 0:
                G.eval()
                f_imgs_sample = (G(z_sample).data + 1) / 2.0
                save_dir = args.sample_dir
                utils.mkdir(save_dir)
                torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)
    
        utils.save_checkpoint({'epoch': epoch + 1,
                               'D': D.state_dict(),
                               'G': G.state_dict(),
                               'd_optimizer': d_optimizer.state_dict(),
                               'g_optimizer': g_optimizer.state_dict()},
                              '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                              max_keep=None)

def sample(G, sample_output_path, sample_num=64):
    G.eval()
    loader = tqdm.tqdm(range(sample_num))
    for sample_idx in loader:
        z_sample = torch.randn(1, args.z_dim).requires_grad_(True)
        z_sample = utils.cuda(z_sample)
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        save_dir = args.sample_dir
        utils.mkdir(sample_output_path)
        torchvision.utils.save_image(f_imgs_sample, '{}/sample_{:d}.jpg'.format(save_dir, sample_idx), nrow=1)



def main(): 
    """ model """
    D = models.DiscriminatorWGANGP(3)
    G = models.Generator(args.z_dim)
    utils.cuda([D, G])
    
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    """ load checkpoint """
    ckpt_dir = args.model_path
    utils.mkdir(ckpt_dir)
    try:
        ckpt = utils.load_checkpoint(ckpt_dir)
        start_epoch = ckpt['epoch']
        D.load_state_dict(ckpt['D'])
        G.load_state_dict(ckpt['G'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    except:
        print(' [*] No checkpoint!')
        start_epoch = 0

    if args.train:
        transform = get_transform_celebA()
        imagenet_data = dsets.ImageFolder(args.data_root, transform=transform)
        data_loader = torch.utils.data.DataLoader(imagenet_data,
                                                  batch_size=args.batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

        train(G, D, g_optimizer, d_optimizer, data_loader, ckpt_dir=ckpt_dir, epochs=args.epochs, start_epoch=start_epoch)
    else: # test
        sample(G, sample_output_path=args.sample_output_path, sample_num=args.sample_num)
    
    
if __name__ == "__main__":
    main()