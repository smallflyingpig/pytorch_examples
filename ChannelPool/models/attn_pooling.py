import torch.nn as nn
import torch.nn.functional as F 
import torch
class AttnPooling(nn.Module):
    def __init__(self, input_channel, pooling_kernel, pooling_stride=1, pooling_padding=0):
        super(AttnPooling, self).__init__()
        self.attn_layer = nn.Conv2d(input_channel, 1, 1,1,0, bias=True)
        self.avg_pooling = nn.AvgPool2d(pooling_kernel, pooling_stride, pooling_padding)
    
    def forward(self, x):
        attn_map = self.attn_layer(x)
        x = x*attn_map
        x = self.avg_pooling(x)
        return x