import torch.nn as nn
import torch
import torch.nn.functional as F

class Revert(nn.Module):
    def __init__(self, input_channel):
        super(Revert, self).__init__()
        self.bn = nn.BatchNorm2d(input_channel*2)
        self.conv = nn.Conv2d(input_channel*2, input_channel, kernel_size=1, groups=input_channel, bias=False)

    def forward(self, x):
        B,C,H,W = x.size()
        x = x.view(B,1,C,H,W)
        x = torch.cat([x, -x], dim=1).transpose(1,2).contiguous().view(B,C*2, H, W)
        x = self.bn(F.relu(x))
        x = self.conv(x)
        return x