import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

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
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
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


    #---------------------------------
        

        self.c_size = 40
        self.conv_1 = nn.Conv2d(self.c_size, 64, kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(512, 512, kernel_size=1,padding=0)
        self.conv_3 = nn.Conv2d(512, 1024, kernel_size=3,padding=1)
        self.conv_4 = nn.Conv2d(1024, 512, kernel_size=1,padding=0)
        self.conv_5 = nn.Conv2d(512, 256, kernel_size=1,padding=0)
        self.deconv_5 = nn.Upsample(size=[128,128], mode='bilinear',align_corners=False)
        self.conv_6 = nn.Conv2d(512, 256, kernel_size=1,padding=0)
        self.conv_7 = nn.Conv2d(256, 512, kernel_size=3,padding=1)
        self.conv_8 = nn.Conv2d(512, 256, kernel_size=1,padding=0)
        self.conv_9 = nn.Conv2d(256, 256, kernel_size=3,padding=1)
        self.conv_10 = nn.Conv2d(768, 512, kernel_size=1,padding=0)
        #self.res_yolo(img_conv1,64,1)
    def res_yolo(self, inputs, filters, res_num):
            res_con1 = nn.Conv2d(inputs.shape[1], filters,(3, 3),padding=1).cuda()
            inputs = res_con1(inputs)
            maximum_pooling = nn.MaxPool2d(2).cuda()
            inputs = maximum_pooling(inputs)
            # inputs = slim.max_pool2d(inputs, [2, 2])
            #print('inputs_shape = {}'.format(inputs.shape))
            for i in range(res_num):
                shortcut = inputs
                res_con2 = nn.Conv2d(filters, int(filters/2),(1, 1),padding=0).cuda()
                inputs = res_con2(inputs)
                res_con3 = nn.Conv2d(int(filters/2), filters,(3, 3),padding=1).cuda()
                inputs = res_con3(inputs)
                #inputs = slim.conv2d(inputs, filters/2, [1, 1])
                #inputs = slim.conv2d(inputs, filters, [3, 3])
                inputs = inputs + shortcut
            return inputs
    #-----------------------------------------------------
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']

        voxel_coords = data_dict['voxel_coords']

        voxels = data_dict['voxels']
        # print(data_dict)
        # import sys
        # sys.exit()
        # print('voxel_features = {}'.format(voxel_features))
        # print('voxel_coords = {}'.format(voxel_coords))
        import numpy as np
        # print(spatial_features.shape)
        # import sys
        # sys.exit()
        batch_size = data_dict['batch_size']

        h_size = spatial_features.shape[2]*data_dict['spatial_features_stride']
        w_size = spatial_features.shape[3]*data_dict['spatial_features_stride']
        c_size = self.c_size

        # print(batch_size,h_size,w_size,c_size)
        img_coors = voxel_coords.cpu().numpy()[:,[0,1,3,2]] # -->[batch_size,c_size,h_size,w_size]
        # print('img_coor = {}'.format(img_coor))
        img_3d = np.zeros([batch_size,c_size,h_size,w_size])

        for img_coor in img_coors:
            # print(i)
            img_3d[int(img_coor[0])][int(img_coor[1])][int(img_coor[2])][int(img_coor[3])] = 1   # img_3d = [1,1024,1024,40]
        img_3d = torch.from_numpy(img_3d).cuda().float()

        #print('shape = {}'.format(img_3d.shape))
        img_conv1 = self.conv_1(img_3d)
        #print('img_conv1_shape = {}'.format(img_conv1.shape))
        img_conv2 = self.res_yolo(img_conv1,64,1)
        img_conv3 = self.res_yolo(img_conv2,128,2)
        img_conv4 = self.res_yolo(img_conv3,256,4)
        img_conv5 = self.res_yolo(img_conv4,512,6)
        img_conv5 = self.conv_2(img_conv5)
        img_conv5 = self.conv_3(img_conv5)
        img_conv5 = self.conv_4(img_conv5)
        img_conv5 = self.conv_5(img_conv5)
        img_deconv_5 = self.deconv_5 (img_conv5)
        #print('img_conv4_out_shape = {}'.format(img_conv4.shape))
        #print('img_conv5_out_shape = {}'.format(img_deconv_5.shape))
        img_deconv_5 = torch.cat((img_conv4, img_deconv_5), 1)
        #print ('img_conv5.shape = {}'.format(img_conv5.shape))
        img_deconv_5 = self.conv_6(img_deconv_5)
        img_deconv_5 = self.conv_7(img_deconv_5)
        img_deconv_5 = self.conv_8(img_deconv_5)
        feature_out = self.conv_9(img_deconv_5)
        # from PIL import Image
        # im = Image.fromarray(np.uint8(img_2d),'L')
        # im.save('/data/OpenPCDet/123.png')
        # for i in spatial_features:
        #     for j in i:
        #         for k in j:
        #             print(k)

        # print('spatial_features = {}'.format(spatial_features.shape))
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
	
        fusion_feature = torch.cat((x, feature_out), 1)
        feature_out = self.conv_10(fusion_feature)
        data_dict['spatial_features_2d'] = x
        # print('x_shape = {}'.format(x.shape))
        data_dict['spatial_features_2d'] = feature_out
        # print('feature_out_shape = {}'.format(feature_out.shape))
        return data_dict
