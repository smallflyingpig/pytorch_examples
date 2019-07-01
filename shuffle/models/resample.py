import torch
from torch import nn
import torch.nn.functional as F


class ReSampleLayer(nn.Module):
    def __init__(self, in_channel):
        super(ReSampleLayer, self).__init__()
        self.transform = nn.Conv2d(in_channel, 2, 3,1,1, bias=False)
        self.init_weight()

    def init_weight(self):
        self.transform.weight.data.fill_(0.1)

    def get_grid(self, grid):
        B, C, H, W = grid.shape
        grid_y = F.softmax(grid[:,0,:,:], dim=1)
        grid_x = F.softmax(grid[:,1,:,:], dim=2)
        grid_x, grid_y = 1-grid_x, 1-grid_y
        for idx in range(H-1):
            grid_y[:, idx+1] = grid_y[:, idx] + grid_y[:, idx+1]
        grid_y = grid_y /(H-1)*2-1 #[-1,1]
        for idx in range(W-1):
            grid_x[:,:,idx+1] = grid_x[:,:,idx] + grid_x[:,:,idx+1]
        grid_x = grid_x /(W-1)*2-1 #[-1,1]
        grid = torch.stack([grid_y, grid_x], dim=1).permute(0,2,3,1)
        # print(grid.max(), grid.min())

        return grid

    def forward(self, x):
        B, C, H, W = x.shape
        grid = self.transform(x)
        grid = self.get_grid(grid)
        x = F.grid_sample(x, grid)
        return x

        





