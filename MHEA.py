import numpy as np
import torch
from torch import nn
from torch.nn import init
from einops import rearrange


class MultiHeadExternalAttention(nn.Module):
    def __init__(self, channels, S=64, heads=8, shortcut=True, dropout=0.0):
        super().__init__()
        assert channels % heads == 0, 'The number of channels should be divisible by heads.'
        self.heads = heads
        self.shortcut = shortcut
        self.mk=nn.Linear(channels // heads, S, bias=False)
        self.mv=nn.Linear(S, channels // heads, bias=False)
        self.softmax=nn.Softmax(dim=2)
        self.dpr = dropout
        self.dropout = nn.Dropout(self.dpr)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def to_3d(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')
    
    def to_4d(self, x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def forward(self, queries): # B, C, H, W
        assert queries.size(2) == queries.size(3), 'H must == W'
        H = queries.size(2)
        if self.shortcut:
            sc = queries
        else:
            sc = 0
        queries = self.to_3d(queries) # B, N, C
        queries = queries.view(queries.size(0), queries.size(1), self.heads, -1) # B, N, heads, C//heads
        queries = queries.permute(0, 2, 1, 3) # B, heads, N, C//heads
        attn = self.mk(queries) # B, heads, N, S
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=3, keepdim=True) 
        out = self.mv(attn) # B, heads, N, C//heads
        out = out.permute(0, 2, 1, 3) # B, N, heads, C//heads
        # dropout
        if self.dpr > 0:
            out = self.dropout(out)
        out = out.reshape(out.size(0), out.size(1), -1) # B, N, C
        out = self.to_4d(out, h=H, w=H)
        out = out + sc
        return out
