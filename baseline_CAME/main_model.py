import torch
import torch.nn as nn

from LayerNorm import LayerNorm
from local_block import Local_block
from global_block import BasicLayer, PatchEmbed, PatchMerging
from EA import ExternalAttention
from MSFA import MSFA

class main_model(nn.Module):

    def __init__(self, num_classes, patch_size=4, in_chans=3, embed_dim=96, depths=(2, 2, 2, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, qkv_bias=True, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0., norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, HFF_dp=0.,
                 conv_depths=(2, 2, 2, 2), conv_dims=(96, 192, 384, 768), conv_drop_path_rate=0.,
                 conv_head_init_scale: float = 1., **kwargs):
        super().__init__()


        ####################################
        ###### Local Branch Setting  #######
        ####################################

        # stem + 3 stage downsample
        self.downsample_layers = nn.ModuleList() 

        ## stem: stem + LN
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        ## stage2-4 downsample：LN + Conv2d
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer) 
              
        # 4 feature resolution stages, each consisting of multiple blocks
        self.stages = nn.ModuleList()  
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        ## Build stacks of local blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(conv_depths[i])]
            )
            self.stages.append((stage))
            cur += conv_depths[i]


        ####################################
        ###### Global Branch Setting  ######
        ####################################

        self.num_layers = len(depths)
        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 4 stage downsample：PatchMerging + global_block
        self.basic_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.basic_layers.append(BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    window_size=window_size,
                                    qkv_bias=qkv_bias,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer > 0) else None,
                                    use_checkpoint=use_checkpoint))


        ######################################
        ###### Prediction Head Setting #######
        ######################################

        self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # final norm layer
        self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        self.conv_head.weight.data.mul_(conv_head_init_scale)
        self.conv_head.bias.data.mul_(conv_head_init_scale)


        ###### init_weights ######
        self.apply(self._init_weights)


        #########################################
        ###### External Attention Setting  ######
        #########################################

        self.ea = nn.ModuleList()
        self.EA_1 = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                                            ExternalAttention(channels=conv_dims[0]))
        self.ea.append(self.EA_1)
        for i in range(3):
            EA_layer = nn.Sequential(nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2),
                                    ExternalAttention(channels=conv_dims[i+1]))
            self.ea.append(EA_layer)


        ############################################
        ######  Feature Fusion Block Setting #######
        ############################################

        self.fu1 = MSFA(ch_local=conv_dims[0], ch_global=conv_dims[0], ch_ea=conv_dims[0], ch_prelayer=conv_dims[0]//2, 
                        ch_inter=conv_dims[0], ch_out=conv_dims[0], drop_rate=HFF_dp)
        self.fu2 = MSFA(ch_local=conv_dims[1], ch_global=conv_dims[1], ch_ea=conv_dims[1], ch_prelayer=conv_dims[1]//2, 
                        ch_inter=conv_dims[1], ch_out=conv_dims[1], drop_rate=HFF_dp)
        self.fu3 = MSFA(ch_local=conv_dims[2], ch_global=conv_dims[2], ch_ea=conv_dims[2], ch_prelayer=conv_dims[2]//2,
                        ch_inter=conv_dims[2], ch_out=conv_dims[2], drop_rate=HFF_dp)
        self.fu4 = MSFA(ch_local=conv_dims[3], ch_global=conv_dims[3], ch_ea=conv_dims[3], ch_prelayer=conv_dims[3]//2,
                        ch_inter=conv_dims[3], ch_out=conv_dims[3], drop_rate=HFF_dp)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, imgs):

        ############################
        ######  Global Branch ######
        ############################

        x_s, H, W = self.patch_embed(imgs)
        x_s = self.pos_drop(x_s)
        x_s_1, H, W = self.basic_layers[0](x_s, H, W)
        x_s_2, H, W = self.basic_layers[1](x_s_1, H, W)
        x_s_3, H, W = self.basic_layers[2](x_s_2, H, W)
        x_s_4, H, W = self.basic_layers[3](x_s_3, H, W)

        # [B,L,C] ---> [B,C,H,W]
        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(x_s_1.shape[0], -1, 56, 56)
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], -1, 28, 28)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], -1, 14, 14)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], -1, 7, 7)

        ###########################
        ######  Local Branch ######
        ###########################

        x_c = self.downsample_layers[0](imgs)
        x_c_1 = self.stages[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)

        ########################################
        ######  External Attention Branch ######
        ########################################

        x_a_1 = self.ea[0](imgs)
        x_a_2 = self.ea[1](x_a_1)
        x_a_3 = self.ea[2](x_a_2)
        x_a_4 = self.ea[3](x_a_3)

        ##################################
        ######  Feature Fusion Path ######
        ##################################

        x_f_1 = self.fu1(x_c_1, x_s_1, x_a_1, None)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_a_2, x_f_1)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_a_3, x_f_2)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_a_4, x_f_3)

        ##############################
        ######  Prediction Head ######
        ##############################

        x_fu = self.conv_norm(x_f_4.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x_fu = self.conv_head(x_fu)

        return x_fu


def CAME(num_classes: int):
    model = main_model(depths=(2, 2, 2, 2),
                     conv_depths=(2, 2, 2, 2),
                     num_classes=num_classes)
    return model
