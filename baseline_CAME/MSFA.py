import torch
import torch.nn as nn

from DropPath import DropPath
from LayerNorm import LayerNorm
from Conv import Conv


class MSFA(nn.Module):
    def __init__(self, ch_local, ch_global, ch_ea, ch_prelayer, ch_inter, ch_out, drop_rate=0.):
        r'''
        ch_local: local feature input dim
        ch_global: global feature input dim
        ch_ea: external attention input dim
        '''
        super(MSFA, self).__init__()

        self.W_l = Conv(ch_local, ch_inter, 1, bn=True, relu=False) # bn
        self.W_g = Conv(ch_global, ch_inter, 1, bn=True, relu=False) # bn
        self.W_ea = Conv(ch_ea, ch_inter, 1, bn=True, relu=False) # bn
        self.W_pre = Conv(ch_prelayer, ch_inter, 1, bn=True, relu=True) # ch_prelayer == ch_inter/2

        self.sa_l = SpatialAttention(kernel_size=3)
        self.sa_g = SpatialAttention(kernel_size=3)
        self.sa_ea = SpatialAttention(kernel_size=3)
        self.ca_l = ChannelAttention(in_planes=ch_inter)
        self.ca_g = ChannelAttention(in_planes=ch_inter)
        self.ca_ea = ChannelAttention(in_planes=ch_inter)

        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm_all = LayerNorm(ch_inter * 4, eps=1e-6, data_format="channels_first")
        self.dowm_all = Conv(ch_inter * 4, ch_inter, 1, bn=True, relu=False)
        self.norm_all_no_pre = LayerNorm(ch_inter * 3, eps=1e-6, data_format="channels_first")
        self.dowm_all_no_pre = Conv(ch_inter * 3, ch_inter, 1, bn=True, relu=False)
        self.norm_all_end = LayerNorm(ch_inter * 4, eps=1e-6, data_format="channels_first")
        self.irmlp = IRMLP(ch_inter * 4, ch_out)
        self.gelu = nn.GELU()

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, ea, p):

        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g)   # global feature from Global Feature Block
        W_EA = self.W_ea(ea)   # external attention

        # previous layer fuse 
        if p is not None:
            W_PRE = self.W_pre(p)
            W_PRE = self.Avg(W_PRE)
            shortcut = W_PRE
            pre_fuse = torch.cat([W_local, W_global, W_EA, W_PRE], 1)
            pre_fuse = self.norm_all(pre_fuse)
            pre_fuse = self.dowm_all(pre_fuse)
            pre_fuse = self.gelu(pre_fuse)
        else:
            shortcut = 0
            pre_fuse = torch.cat([W_local, W_global, W_EA], 1)
            pre_fuse = self.norm_all_no_pre(pre_fuse)
            pre_fuse = self.dowm_all_no_pre(pre_fuse)
            pre_fuse = self.gelu(pre_fuse)

        # local feature
        l_jump_1 = l
        l = self.sa_l(l)
        l = l_jump_1 * l
        l_jump_2 = l
        l = self.ca_l(l)
        l = l_jump_2 * l

        # global feature
        g_jump_1 = g
        g = self.sa_g(g)
        g = g_jump_1 * g
        g_jump_2 = g
        g = self.ca_g(g)
        g = g_jump_2 * g

        # external attention
        ea_jump_1 = ea
        ea = self.sa_ea(ea)
        ea = ea_jump_1 * ea
        ea_jump_2 = ea
        ea = self.ca_ea(ea)
        ea = ea_jump_2 * ea

        # fuse_all
        fuse = torch.cat([l, g, ea, pre_fuse], 1)
        fuse = self.norm_all_end(fuse)
        fuse = self.irmlp(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse


#### Inverted Residual MLP
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
