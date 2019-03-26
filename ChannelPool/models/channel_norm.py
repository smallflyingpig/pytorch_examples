import torch.nn as nn
import torch.nn.functional as F 
import torch
class ChannelNorm(nn.Module):
    def __init__(self, channel, affine=True):
        super(ChannelNorm, self).__init__()
        self.affine = affine
        self.param = nn.Parameter(torch.zeros(channel))
    
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B,C,H*W)
        mean = x.mean(dim=2).view(B,C,1).repeat(1,1,H*W)
        x = (x-mean)
        if self.affine:
            std = x.std(dim=2).view(B,C,1).repeat(1,1,H*W)
            x = x - self.param.view(1,1,H*W)*std
        x = x.view(B,C,H,W) 

        return x