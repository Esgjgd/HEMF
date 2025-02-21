import torch
import torch.nn as nn

from DropPath import DropPath
from LayerNorm import LayerNorm


# Enhanced Local Feature Block
class ELF_block(nn.Module):
    '''
    dim (int): Number of input channels.
    drop_rate (float): Stochastic depth rate. Default: 0.0
    '''
    def __init__(self, dim, drop_rate=0.):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)  # 3x3 conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # 3x3 conv
        x = self.conv(x)
        x = self.norm(x)
        # depthwise conv
        x = self.dwconv(x)
        # pointwise/1x1 conv
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.pwconv(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        # GELU
        x = self.act(x)
        x = shortcut + self.drop_path(x)
        return x
