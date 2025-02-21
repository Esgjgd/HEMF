import numpy as np
import torch
from torch import nn
from torch.nn import init
from einops import rearrange
from LayerNorm import LayerNorm


class ExternalAttention(nn.Module):
    def __init__(self, channels, S=64):
        super().__init__()
        self.ln = LayerNorm(channels, eps=1e-6, data_format="channels_first")
        self.mk = nn.Linear(channels, S,bias=False)
        self.mv = nn.Linear(S, channels,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def to_3d(self, x):
        return rearrange(x, 'b c h w -> b (h w) c')
    
    def to_4d(self, x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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

    def forward(self, queries): # B, C, H, W
        assert queries.size(2) == queries.size(3), 'H must == W'
        H =  queries.size(2)

        shortcut = queries
        queries = self.ln(queries) # B, C, H, W
        queries = self.to_3d(queries) # B, N, C
        attn=self.mk(queries)

        # Norm
        attn = self.softmax(attn) 
        attn = attn / torch.sum(attn,dim=2,keepdim=True)

        out = self.mv(attn)
        out = self.to_4d(out, h=H, w=H) # B, C, H, W
        out = out + shortcut

        return out
