import torch.nn as nn
import torch.nn.functional as F 
class ChannelNorm(nn.Module):
    def __init__(self, channel, affine=True):
        super(ChannelNorm, self).__init__()
        self.affine = affine
        self.instance_norm = nn.InstanceNorm1d(channel, affine=affine)
    
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2,3, 1).view(B, H*W, C)
        
        x = self.instance_norm(x)
        x = x.view(B, H, W, C).permute(0,3,1,2)

        return x