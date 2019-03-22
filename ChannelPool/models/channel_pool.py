
import torch.nn as nn
class GroupChannelPooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(GroupChannelPooling, self).__init__()
        self.pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()
        x = x.view(B, C, H*W).transpose(1,2)
        x = self.pooling(x)
        x = x.transpose(1,2).view(B, -1, H, W)
        return x