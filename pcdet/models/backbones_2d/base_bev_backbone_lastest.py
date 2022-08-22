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
        input_channels = 512
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

        #--------------------------------------------------
        self.c_size = 40
        self.pv_conv_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.01),
            nn.ReLU()
        )
        self.pv_conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            nn.ReLU()
            )
        # self.pv_conv_1 = nn.Conv2d(self.c_size, 64, kernel_size=3,padding=1)

        # self.pv_blocks = nn.ModuleList()
        self.pv_blocks_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False,stride=2),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
            )
        self.pv_blocks_2 = nn.Sequential(       
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False,stride=2),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        self.pv_blocks_3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False,stride=2),
                nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        
        self.pv_out_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )

        # cur_layers = []
        # for k in range(3):
        #     each_layers.extend([
        #         nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #         nn.ReLU()
        #     ])
        # self.pv_blocks.append(nn.Sequential(*cur_layers))
    # #-------------------------------------------------------------------------------------------- 
        self.c_size = 3

        self.conv_1 = nn.Sequential(
                nn.Conv2d(self.c_size, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_2 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_3 = nn.Sequential(
                nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(1024, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_4 = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_5 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.deconv_5 = nn.Upsample(size=[128,128], mode='bilinear',align_corners=False)
        self.conv_6 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_7 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_8 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU())
        self.conv_9 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU())
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
        # self.res_yolo(img_conv1,64,1)
            # self.res_yolo_1_1 = nn.Sequential(
            #     nn.Conv2d(inputs.shape[1], filters, kernel_size=3, padding=1, bias=False).cuda(),
            #     nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01).cuda(),
            #     nn.ReLU().cuda()
            #     nn.MaxPool2d(2).cuda()
            #     )
            # self.res_yolo_1_2  = nn.ModuleList()
            # each_layers = []
            # for k in range(1):
            #     each_layers.extend([  
            #     nn.Conv2d(filters, int(filters/2), kernel_size=1, padding=0, bias=False).cuda(),
            #     nn.BatchNorm2d(int(filters/2), eps=1e-3, momentum=0.01).cuda(),
            #     nn.ReLU().cuda(),          
            #     nn.Conv2d(int(filters/2),filters, kernel_size=3, padding=1, bias=False).cuda(),
            #     nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01).cuda(),
            #     nn.ReLU().cuda()
            #     ])
            # self.res_yolo_1_2 = nn.Sequential(
            #     nn.Conv2d(inputs.shape[1], filters, kernel_size=3, padding=1, bias=False).cuda(),
            #     nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01).cuda(),
            #     nn.ReLU().cuda()
            #     nn.MaxPool2d(2).cuda()
            #     )
    def res_yolo(self, inputs, filters, res_num):
        res_con1 = nn.Sequential(
        nn.Conv2d(inputs.shape[1], filters, kernel_size=3, padding=1, bias=False).cuda(),
        nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01).cuda(),
        nn.ReLU().cuda())

        # res_con1 = nn.Conv2d(inputs.shape[1], filters,(3, 3),padding=1).cuda()
        inputs = res_con1(inputs)
        maximum_pooling = nn.MaxPool2d(2).cuda()
        inputs = maximum_pooling(inputs.contiguous())
        # inputs = slim.max_pool2d(inputs, [2, 2])
        #print('inputs_shape = {}'.format(inputs.shape))
        for i in range(res_num):
           shortcut = inputs
           res_con2 = nn.Sequential(
           nn.Conv2d(filters, int(filters/2), kernel_size=1, padding=0, bias=False).cuda(),
           nn.BatchNorm2d(int(filters/2), eps=1e-3, momentum=0.01).cuda(),
           nn.ReLU().cuda())
           # res_con2 = nn.Conv2d(filters, int(filters/2),(1, 1),padding=0).cuda()
           inputs = res_con2(inputs)
           res_con3 = nn.Sequential(
           nn.Conv2d(int(filters/2),filters, kernel_size=3, padding=1, bias=False).cuda(),
           nn.BatchNorm2d(filters, eps=1e-3, momentum=0.01).cuda(),
           nn.ReLU().cuda())
           # res_con3 = nn.Conv2d(int(filters/2), filters,(3, 3),padding=1).cuda()
           inputs = res_con3(inputs)
           #inputs = slim.conv2d(inputs, filters/2, [1, 1])
           #inputs = slim.conv2d(inputs, filters, [3, 3])
           inputs = inputs + shortcut
        return inputs
        # # --------------------------------------------------------------------------------------
    def livox_network(self,img_3d):
         img_conv1 = self.conv_1(img_3d)
         img_conv2 = self.res_yolo(img_conv1,64,1)
         img_conv3 = self.res_yolo(img_conv2,128,2)
         img_conv4 = self.res_yolo(img_conv3,256,4)
         img_conv5 = self.res_yolo(img_conv4,512,6)
         img_conv5 = self.conv_2(img_conv5)
         img_conv5 = self.conv_3(img_conv5)
         img_conv5 = self.conv_4(img_conv5)
         img_conv5 = self.conv_5(img_conv5)
         img_deconv_5 = self.deconv_5 (img_conv5)
         img_deconv_5 = torch.cat((img_conv4, img_deconv_5), 1)
         img_deconv_5 = self.conv_6(img_deconv_5)
         img_deconv_5 = self.conv_7(img_deconv_5)
         img_deconv_5 = self.conv_8(img_deconv_5)
         feature_out = self.conv_9(img_deconv_5)
         return feature_out

    def pvrcnn_network(self,img_3d):
        img_conv1 = self.pv_conv_1(img_3d)
        img_conv2 = self.pv_conv_2(img_conv1)
        img_conv3 = self.pv_blocks_1(img_conv2)
        img_conv4 = self.pv_blocks_2(img_conv3)
        img_conv5 = self.pv_blocks_3(img_conv4)
        out = self.pv_out_conv(img_conv5)
        return out

    #--------------pillars-----------
        
        # if use_bev:
        #     self.bev_extractor = Sequential(
        #         Conv2d(6, 32, 3, padding=1),
        #         BatchNorm2d(32),
        #         nn.ReLU(),
        #         # nn.MaxPool2d(2, 2),
        #         Conv2d(32, 64, 3, padding=1),
        #         BatchNorm2d(64),
        #         nn.ReLU(),
        #         nn.MaxPool2d(2, 2),
        #     )
        #     block2_input_filters += 64
        p_layer_nums = [3, 5, 5]
        p_layer_strides =  [2, 2, 2]
        p_num_filters = [64, 128, 256]
        p_upsample_strides = [1, 2, 4]
        p_num_upsample_filters= [128, 128, 128]
        block2_input_filters = p_num_filters[0]

        # self.p_block1 = nn.Sequential(
        #         nn.ZeroPad2d(1),
        #         nn.Conv2d(
        #             40, p_num_filters[0], 3, stride=p_layer_strides[0]),
        #         nn.BatchNorm2d(p_num_filters[0]),
        #         nn.ReLU(),
        #     )
        p_cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(40, p_num_filters[0], 3, stride=p_layer_strides[0]),
            nn.BatchNorm2d(p_num_filters[0]),
            nn.ReLU(),
            ]
        for i in range(p_layer_nums[0]):
            each_layers = [
                nn.Conv2d(p_num_filters[0], p_num_filters[0], 3, padding=1),
                nn.BatchNorm2d(p_num_filters[0]),
                nn.ReLU()
                ]
            p_cur_layers = p_cur_layers + each_layers
        self.p_block1 = nn.Sequential(*p_cur_layers)
            # self.p_block1.add(
            #     nn.Conv2d(p_num_filters[0], p_num_filters[0], 3, padding=1))
            # self.p_block1.add(nn.BatchNorm2d(p_num_filters[0]))
            # self.p_block1.add(nn.ReLU())


        self.p_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                p_num_filters[0],
                p_num_upsample_filters[0],
                p_upsample_strides[0],
                stride=p_upsample_strides[0]),
            nn.BatchNorm2d(p_num_upsample_filters[0]),
            nn.ReLU(),
        )

        p_cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(
                block2_input_filters,
                p_num_filters[1],
                3,
                stride=p_layer_strides[1]),
            nn.BatchNorm2d(p_num_filters[1]),
            nn.ReLU(),
            ]
        # self.p_block2 = nn.Sequential(
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(
        #         block2_input_filters,
        #         p_num_filters[1],
        #         3,
        #         stride=p_layer_strides[1]),
        #     nn.BatchNorm2d(p_num_filters[1]),
        #     nn.ReLU(),
        # )
        for i in range(p_layer_nums[1]):
            each_layers = [
                nn.Conv2d(p_num_filters[1], p_num_filters[1], 3, padding=1),
                nn.BatchNorm2d(p_num_filters[1]),
                nn.ReLU()
                ]
            p_cur_layers = p_cur_layers + each_layers
            # self.p_block2.add(
            #     nn.Conv2d(p_num_filters[1], p_num_filters[1], 3, padding=1))
            # self.p_block2.add(nn.BatchNorm2d(p_num_filters[1]))
            # self.p_block2.add(nn.ReLU())
        self.p_block2 = nn.Sequential(*p_cur_layers)
        self.p_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                p_num_filters[1],
                p_num_upsample_filters[1],
                p_upsample_strides[1],
                stride=p_upsample_strides[1]),
            nn.BatchNorm2d(p_num_upsample_filters[1]),
            nn.ReLU(),
        )

        p_cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(p_num_filters[1], p_num_filters[2], 3, stride=p_layer_strides[2]),
            nn.BatchNorm2d(p_num_filters[2]),
            nn.ReLU(),
            ]
        # self.p_block3 = nn.Sequential(
        #     nn.ZeroPad2d(1),
        #     nn.Conv2d(p_num_filters[1], p_num_filters[2], 3, stride=p_layer_strides[2]),
        #     nn.BatchNorm2d(p_num_filters[2]),
        #     nn.ReLU(),
        # )
        for i in range(p_layer_nums[2]):
            each_layers = [
                nn.Conv2d(p_num_filters[2], p_num_filters[2], 3, padding=1),
                nn.BatchNorm2d(p_num_filters[2]),
                nn.ReLU()
                ]
            p_cur_layers = p_cur_layers + each_layers
            # self.p_block3.add(
            #     nn.Conv2d(p_num_filters[2], p_num_filters[2], 3, padding=1))
            # self.p_block3.add(nn.BatchNorm2d(p_num_filters[2]))
            # self.p_block3.add(nn.ReLU())
        self.p_block3 = nn.Sequential(*p_cur_layers)
        self.p_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                p_num_filters[2],
                p_num_upsample_filters[2],
                p_upsample_strides[2],
                stride=p_upsample_strides[2]),
            nn.BatchNorm2d(p_num_upsample_filters[2]),
            nn.ReLU(),
        )

        p_cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(384, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ]
        for i in range(1):
            each_layers = [
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
                ]
            p_cur_layers = p_cur_layers + each_layers
        self.p_block4 = nn.Sequential(*p_cur_layers)
        p_cur_layers = [
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            ]
        for i in range(1):
            each_layers = [
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
                ]
            p_cur_layers = p_cur_layers + each_layers
        self.p_block5 = nn.Sequential(*p_cur_layers)
    def pillars_network(self,x):
        x = self.p_block1(x)
        up1 = self.p_deconv1(x)
        # if self._use_bev:
        #     bev[:, -1] = torch.clamp(
        #         torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
        #     x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.p_block2(x)
        up2 = self.p_deconv2(x)
        x = self.p_block3(x)
        up3 = self.p_deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        x = self.p_block4(x)
        x = self.p_block5(x)
        return x
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

        map_data = data_dict['map_data']
        # print('map_data = {}'.format(data_dict['map_data']))
        # print(data_dict)
        # import sys
        # sys.exit()
        # print('voxel_features = {}'.format(voxel_features))
        # print('voxel_coords = {}'.format(voxel_coords))
        # count_num = 0
        # new_tensor = []
        # h_size = 1024
        # w_size = 1024
        # import numpy as np
        # img_2d = np.zeros([h_size,w_size,3])
        # 
        # # print(voxel_coords)
        # from PIL import Image
        # import numpy as np
        # for i,voxel_coord in enumerate(voxel_coords):

        #     map_pixel = map_data[int(voxel_coord[0]),int(voxel_coord[2]),int(voxel_coord[3])]
        #     # map_pixel = map_data[0,int(voxel_coord[1]),int(voxel_coord[0])]
        #     
        #     # c_size = self.c_size
        #     # img_coors = voxel_coords.cpu().numpy()[:,[0,1,3,2]] # -->[batch_size,c_size,h_size,w_size]

        #     
        #    
        #     img_2d[int(voxel_coord[2])][int(voxel_coord[3]),:] = [255,255,255 ]
        #     # img_3d = torch.from_numpy(img_3d).cuda().float()
        #     
        #     if int(map_pixel[0]) ==166 and int(map_pixel[1]) ==206 and int(map_pixel[2]) ==227 :
        #         img_2d[int(voxel_coord[2])][int(voxel_coord[3]),:] = [166,206,227 ]
        #     from PIL import Image
        # im = Image.fromarray(np.uint8(img_2d))
        # im.save('/home/950154_customer/david/OpenPCDet/124.png')
        #----------------------------------------------------------------------------------------------------------
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
        
        #for img_coor in img_coors:
        #    # print(i)
        #    img_3d[int(img_coor[0])][int(img_coor[1])][int(img_coor[2])][int(img_coor[3])] = 1   # img_3d = [1,1024,1024,40]
        #img_3d = torch.from_numpy(img_3d).cuda().float()

        # feature_out = self.livox_network(img_3d)
        # feature_out = self.pillars_network(img_3d)
        # print(feature_out.shape)
        map_data = map_data.permute(0,3,1,2)
        # feature_out = self.livox_network(map_data)
        # print(map_data.shape)
        # print(map_data)
        feature_out = self.pvrcnn_network(map_data)
        #print('feature_out_shape = {}'.format(feature_out.shape))

        #print('spatial_features_shape = {}'.format(spatial_features.shape))
        #import sys
        #sys.exit()
        #-------------------------------------------------------------------------------------------
        # from PIL import Image
        # im = Image.fromarray(np.uint8(img_2d),'L')
        # im.save('/data/OpenPCDet/123.png')
        # for i in spatial_features:
        #     for j in i:
        #         for k in j:
        #             print(k)

        # print('spatial_features = {}'.format(spatial_features.shape))
        x = torch.cat((spatial_features, feature_out), 1)
        ups = []
        ret_dict = {}
        # x = spatial_features
        # x = feature_out
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

        data_dict['spatial_features_2d'] = x
        #print('x_shape = {}'.format(x.shape))
        # data_dict['spatial_features_2d'] = feature_out
        #print('feature_out_shape = {}'.format(feature_out.shape))
        return data_dict
