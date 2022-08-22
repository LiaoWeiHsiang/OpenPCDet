from functools import partial

import spconv
import torch
import torch.nn as nn

from ...utils import common_utils
from .spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out



class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        # de_norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')
        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )

        self.to_2d__block_3 = nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(82, 32, kernel_size=3,stride=1, padding=1, bias=False),                                                                                                                    
                        nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.to_2d__block_4 = nn.Conv2d(
            32, 2,
            kernel_size=1
        )  
        self.unet_out_1 = block(16, 8, 3,norm_fn=norm_fn, indice_key='subm0')
        self.unet_out_2 = block(8, 2, 1,norm_fn=norm_fn, indice_key='subm0')
        # self.unet_out_1 = spconv.SubMConv3d(16, 2, 1, indice_key="subm00")
        self.num_point_features = 16
        
        # self.my_deblock_1 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        # self.my_deblock_2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv') 
        
        # self.my_deblock_3 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv') 
   
        # self.my_deblock_4 = block(16, 16, 3, norm_fn=norm_fn, indice_key='subm1', conv_type='inverseconv') 



        # ======================unet for maps=========================
        self.kernel_size_maps = 5
        self.padding_maps = 2
        self.conv_input_maps = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, self.kernel_size_maps, padding=self.padding_maps, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1_maps = spconv.SparseSequential(
            block(16, 16, self.kernel_size_maps, norm_fn=norm_fn, padding=self.padding_maps, indice_key='subm1'),
        )

        self.conv2_maps = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, self.kernel_size_maps, norm_fn=norm_fn, stride=2, padding=self.padding_maps, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3_maps = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, self.kernel_size_maps, norm_fn=norm_fn, stride=2, padding=self.padding_maps, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4_maps = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, self.kernel_size_maps, norm_fn=norm_fn, stride=2, padding=(0, self.padding_maps, self.padding_maps), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        # if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
        #     last_pad = self.model_cfg.get('last_pad', 0)

        #     self.conv_out = spconv.SparseSequential(
        #         # [200, 150, 5] -> [200, 150, 2]
        #         spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
        #                             bias=False, indice_key='spconv_down2'),
        #         norm_fn(128),
        #         nn.ReLU(),
        #     )
        # else:
        #     self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4_maps = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4_maps = block(128, 64, self.kernel_size_maps, norm_fn=norm_fn, padding=self.padding_maps, indice_key='subm4')
        self.inv_conv4_maps = block(64, 64, self.kernel_size_maps, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3_maps = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3_maps = block(128, 64, self.kernel_size_maps, norm_fn=norm_fn, padding=self.padding_maps, indice_key='subm3')
        self.inv_conv3_maps = block(64, 32, self.kernel_size_maps, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2_maps = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2_maps = block(64, 32, self.kernel_size_maps, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2_maps = block(32, 16, self.kernel_size_maps, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1_maps = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1_maps = block(32, 16, self.kernel_size_maps, norm_fn=norm_fn, indice_key='subm1')
        self.conv5_maps = spconv.SparseSequential(
            block(16, 16, self.kernel_size_maps, norm_fn=norm_fn, padding=self.padding_maps, indice_key='subm1')
        )

        # self.for_cat_tmp = SparseBasicBlock(64, 128, indice_key='subm1', norm_fn=norm_fn)
        self.cat_conv = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )


        self.to_2d__block_3_maps = nn.Sequential(                                                                                                                                                                
                        nn.Conv2d(82, 32, kernel_size=self.kernel_size_maps,stride=1, padding=self.padding_maps, bias=False),                                                                                                                    
                        nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),                                                                                                                                        
                        nn.ReLU()                                                                                                                                                                           
            )
        self.to_2d__block_4_maps = nn.Conv2d(
            32, 2,
            kernel_size=1
        )  
        self.unet_out_1_maps = block(16, 8, 3,norm_fn=norm_fn, indice_key='subm0')
        self.unet_out_2_maps = block(8, 2, 1,norm_fn=norm_fn, indice_key='subm0')
        # self.unet_out_1 = spconv.SubMConv3d(16, 2, 1, indice_key="subm00")
        # self.num_point_features = 16
    
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']

        # gnd_voxel_features, gnd_voxel_coords = batch_dict['gnd_voxel_features'], batch_dict['gnd_voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # if self.conv_out is not None:
        #     # for detection head
        #     # [200, 176, 5] -> [200, 176, 2]
        #     out = self.conv_out(x_conv4)
        #     batch_dict['encoded_spconv_tensor'] = out
        #     batch_dict['encoded_spconv_tensor_stride'] = 8

        #     batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })


        # # for segmentation head
        # # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # # [800, 704, 21] <- [400, 352, 11]ur
        # x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # # [1600, 1408, 41] <- [800, 704, 21]
        # x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # # [1600, 1408, 41] <- [1600, 1408, 41]
        # x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)
        
        # x_up1 = self.unet_out_1(x_up1)
        # x_up1 = self.unet_out_2(x_up1)
        # spatial_features = x_up1.dense()
        # # print(spatial_features.shape)
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)
        
        # # unet_out_2d = self.to_2d__block_1(spatial_features)
        # # unet_out_2d = self.to_2d__block_2(unet_out_2d)
        # unet_out_2d = self.to_2d__block_3(spatial_features)
        # unet_out_2d = self.to_2d__block_4(unet_out_2d)
        # # print(unet_out_2d.size())

        # batch_dict.update({"unet_out_2d":unet_out_2d})




        # ==========================================================

        # voxel_features, voxel_coords = batch_dict['gnd_voxel_features'], batch_dict['gnd_voxel_coords']

        gnd_voxel_features, gnd_voxel_coords = batch_dict['gnd_voxel_features'], batch_dict['gnd_voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor_maps = spconv.SparseConvTensor(
            features=gnd_voxel_features,
            indices=gnd_voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_maps = self.conv_input(input_sp_tensor_maps)
        # print(x.dense().shape)
        x_conv1_maps = self.conv1_maps(x)
        # print(x_conv1.dense().shape)
        x_conv2_maps = self.conv2_maps(x_conv1)
        # print(x_conv2.dense().shape)
        x_conv3_maps = self.conv3_maps(x_conv2)
        # print(x_conv3.dense().shape)
        x_conv4_maps = self.conv4_maps(x_conv3)


        # x_conv4_all = self.for_cat_tmp(x_conv4)

        # x_conv4.features = torch.cat((x_conv4.features, x_conv4_maps.features), dim=1)
        # x_conv4 = self.cat_conv(x_conv4)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)
            batch_dict['encoded_spconv_tensor'] = out
            batch_dict['encoded_spconv_tensor_stride'] = 8

            batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        # print(x_conv4.dense().shape)
        # if self.conv_out is not None:
        #     # for detection head
        #     # [200, 176, 5] -> [200, 176, 2]
        #     out = self.conv_out(x_conv4)
        #     batch_dict['encoded_spconv_tensor'] = out
        #     batch_dict['encoded_spconv_tensor_stride'] = 8

        #     batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        # ==========================================
        # # for segmentation head
        # # [400, 352, 11] <- [200, 176, 5]
        # x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4_maps, self.conv_up_m4_maps, self.inv_conv4_maps)
        # # print(x_up4.dense().shape)
        # # [800, 704, 21] <- [400, 352, 11]ur
        # x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3_maps, self.conv_up_m3_maps, self.inv_conv3_maps)
        # # print(x_up3.dense().shape)
        # # [1600, 1408, 41] <- [800, 704, 21]
        # x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2_maps, self.conv_up_m2_maps, self.inv_conv2_maps)
        # # print(x_up2.dense().shape)
        # # [1600, 1408, 41] <- [1600, 1408, 41]
        # x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1_maps, self.conv_up_m1_maps, self.conv5_maps)
        # # print(x_up1.dense().shape)
        
        # x_up1 = self.unet_out_1_maps(x_up1)
        # x_up1 = self.unet_out_2_maps(x_up1)
        # spatial_features = x_up1.dense()
        # # print(spatial_features.shape)
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)
        
        # # unet_out_2d = self.to_2d__block_1(spatial_features)
        # # unet_out_2d = self.to_2d__block_2(unet_out_2d)
        # unet_out_2d = self.to_2d__block_3_maps(spatial_features)
        # unet_out_2d = self.to_2d__block_4_maps(unet_out_2d)
        # ==========================================
        # batch_dict.update({"unet_out_2d":unet_out_2d})
        return batch_dict
