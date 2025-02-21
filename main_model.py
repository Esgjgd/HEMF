import torch
import torch.nn as nn

from LayerNorm import LayerNorm
from ELF_block import ELF_block
from EGF_block import EGF_block
from HEF_block import HEF_block


class main_model(nn.Module):

    def __init__(self, num_classes, in_chans=3, EGF_depths=(2, 2, 2, 2), ELF_depths=(2, 2, 2, 2), 
                 conv_dims=(96, 192, 384, 768), ELF_drop_path_rate=0.,HEF_drop_rate=0.,
                 conv_head_init_scale: float = 1., **kwargs):
        super().__init__()

        ####################################
        ###### Local Branch Setting  #######
        ####################################

        # stem + 3 stages downsample
        self.downsample_layers = nn.ModuleList() 
        ## stem
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        ## stage2-4 downsample：LN + Conv2d
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                            nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer) 
              
        # 4 feature resolution stages, each consisting of multiple local blocks
        self.stages = nn.ModuleList()  
        dp_rates = [x.item() for x in torch.linspace(0, ELF_drop_path_rate, sum(ELF_depths))]
        cur = 0
        ## Build stacks of local blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[ELF_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
                  for j in range(ELF_depths[i])]
            )
            self.stages.append((stage))
            cur += ELF_depths[i]

        ####################################
        ###### Global Branch Setting  ######
        ####################################

        # stem + 3 stages downsample
        self.downsample_layers2 = nn.ModuleList() 
        ## stem
        stem2 = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers2.append(stem2)
        ## stage2-4 downsample：LN + Conv2d
        for i in range(3):
            downsample_layer2 = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                            nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))

            self.downsample_layers2.append(downsample_layer2) 
            
        # 4 feature resolution stages, each consisting of multiple global blocks
        self.stages2 = nn.ModuleList()  
        ## Build stacks of global blocks in each stage
        for i in range(4):
            stage2 = EGF_block(embed_dim=conv_dims[i], num_heads=8, layer_num = EGF_depths[i])
            self.stages2.append((stage2))

        ######################################
        ###### Prediction Head Setting #######
        ######################################

        self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # final norm layer
        self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        self.conv_head.weight.data.mul_(conv_head_init_scale)
        self.conv_head.bias.data.mul_(conv_head_init_scale)

        ###### init_weights ######
        self.apply(self._init_weights)
        
        ################################################################
        ###### Hierachical Enhanced Feature Fusion Block Setting #######
        ################################################################

        self.fu1 = HEF_block(ch_1=96, ch_2=96, ch_int=96, ch_out=96, drop_rate=HEF_drop_rate)
        self.fu2 = HEF_block(ch_1=192, ch_2=192, ch_int=192, ch_out=192, drop_rate=HEF_drop_rate)
        self.fu3 = HEF_block(ch_1=384, ch_2=384, ch_int=384, ch_out=384, drop_rate=HEF_drop_rate)
        self.fu4 = HEF_block(ch_1=768, ch_2=768, ch_int=768, ch_out=768, drop_rate=HEF_drop_rate)
        
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
        ####### Global Branch ######
        ############################

        x_s = self.downsample_layers2[0](imgs)
        x_s_1 = self.stages2[0](x_s)
        x_s = self.downsample_layers2[1](x_s_1)
        x_s_2 = self.stages2[1](x_s)
        x_s = self.downsample_layers2[2](x_s_2)
        x_s_3 = self.stages2[2](x_s)
        x_s = self.downsample_layers2[3](x_s_3)
        x_s_4 = self.stages2[3](x_s)

        ###########################
        ####### Local Branch ######
        ###########################

        x_c = self.downsample_layers[0](imgs)
        x_c_1 = self.stages[0](x_c)
        x_c = self.downsample_layers[1](x_c_1)
        x_c_2 = self.stages[1](x_c)
        x_c = self.downsample_layers[2](x_c_2)
        x_c_3 = self.stages[2](x_c)
        x_c = self.downsample_layers[3](x_c_3)
        x_c_4 = self.stages[3](x_c)
        
        ######################################################
        ###### Hierachical Enhanced Feature Fusion Path ######
        ######################################################

        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
        
        ##############################
        ####### Prediction Head ######
        ##############################

        x_fu = self.conv_norm(x_f_4.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        x_fu = self.conv_head(x_fu)

        return x_fu



def HEMF(num_classes: int):
    model = main_model(EGF_depths=(3, 3, 3, 3),
                     ELF_depths=(4, 4, 4, 4),
                     num_classes=num_classes)
    return model
