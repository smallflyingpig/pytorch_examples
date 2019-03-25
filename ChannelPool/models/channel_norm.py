import torch.nn as nn
import torch.nn.functional as F 
class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()
    
    def forward(self, x):
        B, C, H, W = x.size()
        x = x.permute(0, 2,3, 1).view(B, H*W, C)
        x = F.layer_norm(x, x.size()[1:])
        x = x.view(B, H, W, C).permute(0,3,1,2)

        return x