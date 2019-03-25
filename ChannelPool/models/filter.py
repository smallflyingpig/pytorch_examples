import torch.nn as nn
import torch
import torch.nn.functional as F

class Filter(nn.Module):
    def __init__(self, filter_num, kernel_size, stride=1, padding=0, bias=False, merge_type='sum'):
        super(Filter, self).__init__()
        self.filters = nn.Conv2d(1, filter_num, kernel_size, stride, padding)
        self.merge_type = merge_type
        self.filter_num = filter_num

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B*C, 1, H, W)
        x = self.filters(x) # (B*C, filter_num, H, W)
        _, filter_num, H, W = x.size()
        x = x.view(B, C, filter_num, H, W)
        if self.merge_type == 'sum':
            x = x.mean(dim=1)
        elif self.merge_type == 'max':
            x = F.max_pool1d(x.permute(0,2,3,4,1).view(B, filter_num*H*W, C), C, 1, padding=0)
            x = x.view(B, filter_num, H, W)
        else:
            raise NotImplementedError
        return x
         
        



