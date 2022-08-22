import numpy as np
import torch
import torch.nn as nn
class ResYolo(nn.Module):
    # expansion = 1
    def __init__(self, in_channels, out_channels, res_num):
        super(ResYolo, self).__init__()
        self.res_num = res_num
        self.block1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )
        self.max_pool = nn.MaxPool2d(2, stride=2)

        self.res_yolo_block = nn.ModuleList()
        for i in range(res_num):   
            cur_layers = [nn.Conv2d(out_channels, int(out_channels/2), kernel_size=1, padding=0, stride=1, bias=False),
                        nn.BatchNorm2d(int(out_channels/2), eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                        nn.Conv2d(int(out_channels/2), out_channels, kernel_size=3, padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ]
            self.res_yolo_block.append(nn.Sequential(*cur_layers))
             
    def forward(self, x):
        x = self.block1(x)
        x = self.max_pool(x)
        
        for i in range(self.res_num):
            shortcut = x
            x = self.res_yolo_block[i](x)
            x = x + shortcut
        
        return x
class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(ResBlock, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm1(self.conv1(x)))
      x = self.relu(self.batch_norm2(self.conv2(x)))
      x = self.batch_norm3(self.conv3(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
    #   print(x.shape)
    #   print(identity.shape)
      x += identity
      x = self.relu(x)
      return x
class GCN(nn.Module):
    def __init__(self,c,out_c,k=(7,7)): #out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0],1), padding =(3,0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1,k[0]), padding =(0,3))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k[1]), padding =(0,3))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1],1), padding =(3,0))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x

class BR(nn.Module):
    def __init__(self, out_c):
        super(BR, self).__init__()
        # self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(out_c,out_c, kernel_size=3,padding=1)
    
    def forward(self,x):
        x_res = self.conv1(x)
        x_res = self.relu(x_res)
        x_res = self.conv2(x_res)
        
        x = x + x_res
        
        return x
class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        # input_channels = 320
        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    ResBlock(num_filters[idx],num_filters[idx]),
                    # ResBlock(num_filters[idx],num_filters[idx])
                    # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=1, padding=0, bias=False),
                    # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    # nn.ReLU(),

                    # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    # nn.ReLU(),

                    # nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=1, padding=0, bias=False),
                    # nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    # nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    # ========================= for seg ================================
        self.unet_block_1_1= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(2, 16, kernel_size=3,stride=2, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.unet_block_1_2= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(16, 16, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.unet_block_1_3= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(16, 16, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        
        self.unet_block_2_1 = nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(16, 32, kernel_size=3,stride=2, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.unet_block_2_2= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(32, 32, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.unet_block_2_3= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(32, 32, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )

        self.unet_block_3_1 = nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(32, 64, kernel_size=3,stride=2, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.unet_block_3_2= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.unet_block_3_3= nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        # self.unet_block_4 = nn.Sequential(                                                                                                                                                                
        #                 nn.Conv2d(64, 128, kernel_size=3,stride=2, padding=1, bias=False),                                                                                                                    
        #                 nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),                                                                                                                                        
        #                 nn.ReLU()                                                                                                                                                                           
        #     )
        # self.gcn1 = GCN(64,64,k=[7,7])
        # self.gcn2 = GCN(128,128,k=[7,7])
        # self.gcn3 = GCN(256,256,k=[7,7])
        # self.br1 = BR(64)
        # self.br2 = BR(128)
        # self.br3 = BR(256)

        # self.br4 = BR(64)
        # self.br5 = BR(128)
        # self.br6 = BR(256)

        # self.gcn_list = [self.gcn1,self.gcn2,self.gcn3]
        # self.br_list = [self.br1,self.br2,self.br3]
        # self.up_br_list = [self.br4,self.br5,self.br6]
        # =======
        # self.gcn1 = GCN(64,64,k=[7,7])
        # self.gcn2 = GCN(128,128,k=[7,7])
        # self.gcn3 = GCN(256,256,k=[7,7])
        # self.gcn4 = GCN(512,512,k=[7,7])
        # self.br1 = BR(64)
        # self.br2 = BR(128)
        # self.br3 = BR(256)
        # self.br4 = BR(512)

        # self.br5 = BR(64)
        # self.br6 = BR(128)
        # self.br7 = BR(256)
        # self.br8 = BR(512)

        # self.gcn_list = [self.gcn1,self.gcn2,self.gcn3,self.gcn4]
        # self.br_list = [self.br1,self.br2,self.br3,self.br4]
        # self.up_br_list = [self.br5,self.br6,self.br7,self.br8]

        # self.first_block = nn.Sequential(
        #         nn.Conv2d(c_in_list[0], 64, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
        #         nn.ReLU())
        # self.blocks = nn.ModuleList()
        # for idx in range(num_levels):
        #     # for k in range(layer_nums[idx]):
        #     if idx ==0:
        #         in_ch = 64
        #     else:
        #         in_ch = c_in_list[idx]
        #     self.blocks.append(nn.Sequential(ResYolo(in_ch,num_filters[idx],layer_nums[idx])))

    def concat_head(self, data_dict):
        spatial_features = data_dict['spatial_features']
        # spatial_features_2d_out = self.deblocks_oo[-1](spatial_features_2d)
        unet_out_2d = data_dict['unet_out_2d']
        unet_out_2d = self.unet_block_1_1(unet_out_2d)
        # unet_out_2d = self.unet_block_1_2(unet_out_2d)
        # unet_out_2d = self.unet_block_1_3(unet_out_2d)
        unet_out_2d = self.unet_block_2_1(unet_out_2d)
        # unet_out_2d = self.unet_block_2_2(unet_out_2d)
        # unet_out_2d = self.unet_block_2_3(unet_out_2d)
        unet_out_2d = self.unet_block_3_1(unet_out_2d)
        # unet_out_2d = self.unet_block_3_2(unet_out_2d)
        # unet_out_2d = self.unet_block_3_3(unet_out_2d)
        # unet_out_2d = self.unet_block_4(unet_out_2d)
        # print(spatial_features.size())
        # print(unet_out_2d.size())
        concat_tensor = torch.cat((spatial_features,unet_out_2d),dim = 1)
        # print(spatial_features_2d_out.size())
        # data_dict["spatial_features_2d_outground_feature_out"] = spatial_features_2d_out
        # concat_tensor = self.concat_block_1(spatial_features_2d_out)        
        # concat_tensor = self.concat_block_2(concat_tensor)
        # concat_tensor = self.concat_block_3(concat_tensor)
        # concat_tensor =  self.concat_block_4(concat_tensor)
            
        # print(concat_tensor.size())
        # print(spatial_features_2d.size())
        # concat_tensor = torch.cat((spatial_features_2d,concat_tensor),dim = 1) 
        data_dict["concat_tensor"] = concat_tensor
        
        # self.forward_ret_dict['ground_feature_out'] = spatial_features_2d_out
        # self.forward_ret_dict['gnd_voxel_coords'] = data_dict["gnd_voxel_coords"]
        

        # gnd_voxel_coords
        # print(gnd_voxel_coords.size())
        return data_dict
    
    # ========================= for seg ================================
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        

        # data_dict = self.concat_head(data_dict)
        # spatial_features = data_dict['concat_tensor']

        ups = []
        ret_dict = {}
        x = spatial_features
        # x = self.first_block(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            # print(x.size())
            # gcn = self.gcn_list[i]
            # br = self.br_list[i]
            # gc_fm = br(gcn(x))
            # print(gc_fm.size())

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
                # cups.append(self.deblocks[i](self.up_br_list[i](gc_fm)))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # print("x_shape = {}".format(x.shape))
        data_dict['spatial_features_2d'] = x

        return data_dict
