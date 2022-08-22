import numpy as np
import torch.nn as nn
import torch
from .anchor_head_template import AnchorHeadTemplate
import spconv

class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # self.deblocks_oo = nn.ModuleList()
        self.deblocks_1 = nn.Sequential(
                        nn.ConvTranspose2d(
                            512, 256,
                            2,
                            2, bias=False
                        ),
                        nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )
        self.deblocks_2 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        self.deblocks_3 = nn.Sequential(
                        nn.ConvTranspose2d(
                            256, 128,
                            2,
                            2, bias=False
                        ),
                        nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )
        self.deblocks_4 = nn.Sequential(
                        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        self.deblocks_5 = nn.Sequential(
                        nn.ConvTranspose2d(
                            128, 64,
                            2,
                            2, bias=False
                        ),
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    )
        self.deblocks_6 = nn.Sequential(
                        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        self.deblocks_7 = nn.Conv2d(
            64, 2,
            kernel_size=1
        )
       
        self.unet_out_1 = spconv.SubMConv3d(16, 2, 1, indice_key="subm00")
       
        self.unet_3d_2d = nn.Conv2d(
            656, 2,
            kernel_size=1
        )

        self.concat_block_1 = nn.Sequential(
                        nn.Conv2d(2, 16, kernel_size=3,stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        self.concat_block_2 = nn.Sequential(
                        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        self.concat_block_3 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        self.concat_block_4 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                        nn.ReLU()
            )
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                512,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)



    def ground_seg_head(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        # spatial_features_2d_out = self.deblocks_oo[-1](spatial_features_2d)

        spatial_features_2d_out = self.deblocks_1(spatial_features_2d)
        spatial_features_2d_out = self.deblocks_2(spatial_features_2d_out)
        spatial_features_2d_out = self.deblocks_3(spatial_features_2d_out)
        spatial_features_2d_out = self.deblocks_4(spatial_features_2d_out)
        spatial_features_2d_out = self.deblocks_5(spatial_features_2d_out)
        spatial_features_2d_out = self.deblocks_6(spatial_features_2d_out)
        spatial_features_2d_out = self.deblocks_7(spatial_features_2d_out)
        # print(spatial_features_2d_out.size())
        # data_dict["spatial_features_2d_outground_feature_out"] = spatial_features_2d_out
        concat_tensor = self.concat_block_1(spatial_features_2d_out)        
        concat_tensor = self.concat_block_2(concat_tensor)
        concat_tensor = self.concat_block_3(concat_tensor)
        # concat_tensor =  self.concat_block_4(concat_tensor)
            
        # print(concat_tensor.size())
        # print(spatial_features_2d.size())
        concat_tensor = torch.cat((spatial_features_2d,concat_tensor),dim = 1) 
        data_dict["concat_tensor"] = concat_tensor
        
        self.forward_ret_dict['ground_feature_out'] = spatial_features_2d_out
        self.forward_ret_dict['gnd_voxel_coords'] = data_dict["gnd_voxel_coords"]
        

        # gnd_voxel_coords
        # print(gnd_voxel_coords.size())
        return data_dict
    def unet_to_2d(self,data_dict):
        unet_out = data_dict["unet_out"]
        unet_out_dense = unet_out.dense()
        N, C, D, H, W = unet_out_dense.shape
        unet_out_dense_2d = unet_out_dense.view(N, C * D, H, W)
        u_out = self.unet_3d_2d (unet_out_dense_2d)
        self.forward_ret_dict['ground_feature_out'] = u_out
        self.forward_ret_dict['gnd_voxel_coords'] = data_dict["gnd_voxel_coords"]
    def unet_3d(self,data_dict):
        unet_out = data_dict["unet_out"]   
        unet_out = self.unet_out_1(unet_out)
        self.forward_ret_dict['unet_out'] = unet_out
        self.forward_ret_dict['gnd_voxel_coords'] = data_dict["gnd_voxel_coords"]
        

    def forward(self, data_dict):

        # print(data_dict["gnd_voxel_coords"].size())
        # print("-----")
        # data_dict = self.ground_seg_head(data_dict)
        # self.forward_ret_dict['unet_out'] = data_dict["unet_out"]
        # self.forward_ret_dict['gnd_voxel_coords'] = data_dict["gnd_voxel_coords"]
        # self.unet_3d(data_dict)
        # self.unet_to_2d(data_dict)
        # self.forward_ret_dict['unet_out'] = data_dict["unet_out"]
        spatial_features_2d = data_dict['spatial_features_2d']
        # print(spatial_features_2d.size())
        # spatial_features_2d = data_dict['concat_tensor']
        # ====
        # self.forward_ret_dict['unet_out_2d'] = data_dict["unet_out_2d"]
        # self.forward_ret_dict['gnd_voxel_coords'] = data_dict["gnd_voxel_coords_GT"]
        # ====

        # spatial_features_2d_out = self.deblocks_oo[-1](spatial_features_2d)
        # print(spatial_features_2d_out.size())

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        return data_dict
