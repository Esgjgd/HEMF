import torch
import torch.nn as nn

from DropPath import DropPath
from LayerNorm import LayerNorm
from Conv import Conv
from MHEA import MultiHeadExternalAttention


# Hierachical Enhanced Feature Fusion Block
class HEF_block(nn.Module):
    '''
    ch_1: local input dim
    ch_2: global input dim
    ch_int: intermediate concat dim, equal to ch_1 and ch_2
    ch_out: output dim
    drop_rate: drop rate
    HEF_block previous layer output dim: ch_int//2
    '''
    def __init__(self, ch_1, ch_2, ch_int, ch_out, drop_rate=0.):
        super(HEF_block, self).__init__()

        # global and local branch 
        ## 1x1 conv
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        ## MEA ln SA CA
        self.norm_l = LayerNorm(ch_2, eps=1e-6, data_format="channels_first")
        self.norm_g = LayerNorm(ch_2, eps=1e-6, data_format="channels_first")
        self.mea_l = MultiHeadExternalAttention(channels=ch_2, shortcut=False)
        self.mea_g = MultiHeadExternalAttention(channels=ch_2, shortcut=False)
        self.sa_l = SpatialAttention(kernel_size=7)
        self.sa_g = SpatialAttention(kernel_size=7)
        self.ca_l = ChannelAttention(in_planes=ch_2, ratio = 16)
        self.ca_g = ChannelAttention(in_planes=ch_2, ratio = 16)

        # previous HEF_block
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.Avg = nn.AvgPool2d(2, stride=2)
        ## pre is not none
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        ## pre is none
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)

        # activation function
        self.gelu = nn.GELU()
        # norm at last
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        # SIRMLP
        self.residual = SIRMLP(ch_1 + ch_2 + ch_int, ch_int, ch_out)
        # drop path
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):
        # from ELF_Block
        W_local = self.W_l(l)
        # from GLF_Block
        W_global = self.W_g(g)
        # previous HEF_block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # local branch
        l = self.norm_l(l)
        ## MEA
        l_jump_0 = l
        l = self.mea_l(l)
        l = l_jump_0 * l
        l = self.norm_l(l)
        ## SA
        l_jump_1 = l
        l = self.sa_l(l)
        l = l * l_jump_1
        ## CA
        l_jump_2 = l
        l = self.ca_l(l)
        l = l * l_jump_2

        # global branch
        g = self.norm_g(g)
        ## MEA
        g_jump_0 = g
        g = self.mea_g(g)
        g = g * g_jump_0
        g = self.norm_g(g)
        ## SA
        g_jump_1 = g
        g = self.sa_g(g)
        g = g * g_jump_1
        ## CA
        g_jump_2 = g
        g = self.ca_g(g)
        g = g * g_jump_2

        # fuse all
        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)

        return fuse


# Squeezed Inverted Residual MLP
class SIRMLP(nn.Module):
    def __init__(self, inp_dim, ch_int, out_dim):
        super(SIRMLP, self).__init__()
        self.conv_s = Conv(inp_dim, ch_int, 1, bn=True, relu=False)
        self.conv1 = Conv(ch_int, ch_int, 3, relu=False, bias=False)
        self.conv2 = Conv(ch_int, ch_int * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(ch_int * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(ch_int)

    def forward(self, x):
        x = self.conv_s(x)
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)
        return out