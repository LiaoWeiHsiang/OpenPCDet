import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
from .focal_v2 import*

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        gamma = 5
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        # print(F.softmax(input))
        logpt = F.log_softmax(input)
        # print(logpt)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict
    
    def get_ground_loss_v3(self):
        unet_out = self.forward_ret_dict['unet_out']
        unet_out_features = unet_out.dense()
        # ground_feature_out = self.forward_ret_dict["ground_feature_out"]
        ground_feature_out = unet_out_features
        gnd_voxel_coords = self.forward_ret_dict["gnd_voxel_coords"]
        # print(type(unet_out_features))
        # print(ground_feature_out.size())
        # print(gnd_voxel_coords.shape)
        # print("---------")
        # print(gnd_voxel_coords)
        # gnd_voxel_coords = np.delete(gnd_voxel_coords, 0, 2)
        # print(gnd_voxel_coords.shape)
        # gt_num = gnd_voxel_coords.shape[1]
        
        labels = np.zeros([len(gnd_voxel_coords),ground_feature_out.size()[2],ground_feature_out.size()[3],ground_feature_out.size()[4]], dtype='float16')
        for batch_num,gnd_voxel_coord in enumerate(gnd_voxel_coords):
            z = gnd_voxel_coord[:,0].astype(int)
            # print(z)
            # print(np.max(z))
            # print(gnd_voxel_coord.shape)
            # print(gnd_voxel_coord.shape)
            y = gnd_voxel_coord[:,1].astype(int)
            x = gnd_voxel_coord[:,2].astype(int)
            # print(y.shape)
            # print(x.shape)
            # print(labels.shape)
            labels[batch_num][z,y,x] = 1
        
        ground_feature_out = ground_feature_out.view(len(gnd_voxel_coords), -1, 2)
        labels = torch.from_numpy(labels).cuda().view(len(gnd_voxel_coords), -1)
        
        # print("where = ")
        
        # print(ground_feature_out.shape)
        # print(labels.shape)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        soft_max = nn.Softmax(dim=1)
        
        for batch_num in range(len(gnd_voxel_coords)):
            
            soft_unet = soft_max(ground_feature_out[batch_num])
            soft_unet_max = torch.argmax(soft_unet, dim=1)
            train_acc = torch.sum(soft_unet_max == labels[batch_num])
            np_soft_unet_max = soft_unet_max.cpu().detach().numpy()
            np_labels = labels[batch_num].cpu().detach().numpy()
            
            # print("np_acc = {}".format(float(len(np.where(np_soft_unet_max==np_labels)[0]))/float(labels[batch_num].size()[0])))
            # print(np.where(np_soft_unet_max==np_labels)[0].shape)
            # print(np.where(np_soft_unet_max==0))
            # print("train_acc = {}".format(train_acc))
            # print("labels[batch_num] = {}".format(labels[batch_num])) 
            # print("soft_unet_max = {}".format(soft_unet_max))    
            # print("---------------------")
            print("acc = {}".format(float(train_acc)/float(labels[batch_num].size()[0])))
            # print("gnd_ratio = {}".format(float(gt_num)/(1024*1024)))
            # focal = FocalLoss(gamma = 5)
            # loss = focal(unet_out_features[batch_num],labels[batch_num].long())
            loss = criterion(ground_feature_out[batch_num], labels[batch_num].long())
            total_loss = total_loss + loss
        # print(gnd_loss)
        return total_loss/len(gnd_voxel_coords) 
 
    def tversky_loss(self, true, logits, alpha, beta, eps=1e-7):
        """Computes the Tversky loss [1].
        Args:
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            alpha: controls the penalty for false positives.
            beta: controls the penalty for false negatives.
            eps: added to the denominator for numerical stability.
        Returns:
            tversky_loss: the Tversky loss.
        Notes:
            alpha = beta = 0.5 => dice coeff
            alpha = beta = 1 => tanimoto coeff
            alpha + beta = 1 => F beta coeff
        References:
            [1]: https://arxiv.org/abs/1706.05721
        """
        num_classes = logits.shape[1]
        # print(logits.size)
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            # true = true.view(-1)
            # print(true.shape)
            # print(true.squeeze(1).shape)
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            
            
            # print(true.size())
            # print(true.squeeze(1).size())
            # print(true_1_hot.shape)
            # print(torch.eye(num_classes))
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            # true_1_hot = true_1_hot.permute(0, 4, 1, 2, 3).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        fps = torch.sum(probas * (1 - true_1_hot), dims)
        fns = torch.sum((1 - probas) * true_1_hot, dims)
        num = intersection
        denom = intersection + (alpha * fps) + (beta * fns)
        tversky_loss = (num / (denom + eps)).mean()
        
        gamma_2 = 0.75
        return (1-tversky_loss)**gamma_2
        # return (1 - tversky_loss)
    def get_ground_loss_v2(self,voxel_coords):                                                                                                                                                                           
                                                                                                                                                             
        ground_feature_out = self.forward_ret_dict["unet_out_2d"]                                                                                                                                    
        gnd_voxel_coords = self.forward_ret_dict["gnd_voxel_coords"]                                                                                                                                        
                                                                                                                                                                   
                 
        ground_label = 0     
        non_ground_label = 1                                                                                                                                                                                  
        labels = np.ones([len(gnd_voxel_coords),ground_feature_out.size()[2],ground_feature_out.size()[3]], dtype='float16')                                                                               
        for batch_num,gnd_voxel_coord in enumerate(gnd_voxel_coords):                                                                                                                                       
                                                                                                                                                                           
            y = gnd_voxel_coord[:,1].astype(int)                                                                                                                                                            
            x = gnd_voxel_coord[:,2].astype(int)                                                                                                                                                            
            labels[batch_num][y,x] = ground_label                                                                                                                                                                  
        
        labels0 = torch.from_numpy(labels).cuda().view(len(gnd_voxel_coords),1,ground_feature_out.size()[2],ground_feature_out.size()[3])
        loss = self.tversky_loss(labels0.long(),ground_feature_out,0.95,0.05)                                                                                                                                                                                                 
        
        # lossHuber = nn.SmoothL1Loss(reduction = "mean").cuda()
        # lossSpatial = SpatialSmoothLoss().cuda()
        

       

        ground_feature_out = ground_feature_out.permute(0,2,3,1)
        ground_feature_out = ground_feature_out.view(len(gnd_voxel_coords), -1,2)                                                                                                                          
        labels = torch.from_numpy(labels).cuda().view(len(gnd_voxel_coords), -1)                                                                                                                            
                                                                                                                                                                                                            
        criterion = nn.CrossEntropyLoss()                                                                                                                                                                   
        total_loss = 0                                                                                                                                                                                      
                                                                                                                                                                                                            
        soft_max = nn.Softmax(dim=1)                                                                                                                                                                        
        for batch_num in range(len(gnd_voxel_coords)):                                                                                                                                                      
            recall_denominator = len(gnd_voxel_coords[batch_num])                                                                                                                                                                   
            soft_unet = soft_max(ground_feature_out[batch_num])                                                                                                                                             
            

            soft_unet_max = torch.argmax(soft_unet, dim=1)                                                                                                                                                  
            train_acc = torch.sum(soft_unet_max == labels[batch_num])                                                                                                                                       
            # np_soft_unet_max = soft_unet_max.cpu().detach().numpy()                                                                                                                                         
            # np_labels = labels[batch_num].cpu().detach().numpy()                                                                                                                                            
            
            # soft_unet_max_for_recall = torch.where(soft_unet_max == non_ground_label, soft_unet_max, 7)
            
          

            # TP = torch.sum(soft_unet_max_for_recall == labels[batch_num])
            

            ground_feature_out_np = soft_unet_max.cpu().detach().numpy() 
            labels_np = labels[batch_num].cpu().detach().numpy() 
            ground_feature_out_np_save = ground_feature_out_np.reshape([1024,1024])
            labels_np_save = labels_np.reshape([1024,1024])
            
            


            ground_feature_out_np = np.where(ground_feature_out_np == non_ground_label, 8, ground_feature_out_np)
            
            molecular = len(np.where(ground_feature_out_np == labels_np)[0])



            equal = np.where(ground_feature_out_np == labels_np)[0]
 

         
            
            pre_gnd_coor = np.where(ground_feature_out_np == ground_label)[0]
            labels_gnd_coor = np.where(labels_np == ground_label)[0]
            
          
            recall = molecular/recall_denominator
            print("recall = {}".format(molecular/recall_denominator))
            
          
            # print("gnd_ratio = {}".format(len(gnd_voxel_coords[batch_num])/len(labels_np)))

            print("precision = {}".format(float(train_acc)/len(labels_np)))      

            # focal = FocalLoss(gamma = 4)                                                                                                                                                                  
            # loss = focal(ground_feature_out[batch_num],labels[batch_num].long())                                                                                                                                                                                      
            # loss = criterion(ground_feature_out[batch_num], labels[batch_num].long())
           

            # gnd_voxel_coords = data_dict["gnd_voxel_coords"]
            # save_path = '/home/950154_customer/david/OpenPCDet/tmp/'
            # name_num = round(recall,2) * 100
            # name = str(name_num)+ "_pre_gnd"
            # import os
            # path = os.path.join(save_path,name+".npy")
            # with open(path, 'wb') as f:
            #     np.save(f, ground_feature_out_np_save)

            # save_path = '/home/950154_customer/david/OpenPCDet/tmp/'
            # # ground_feature_out_np_save
            # name = str(name_num)+ "_label"
            # path = os.path.join(save_path,name+".npy")
            # with open(path, 'wb') as f:
            #     np.save(f, labels_np_save)

            # if torch.is_tensor(voxel_coords):
            #     voxel_coords = voxel_coords.cpu().detach().numpy() 
            # # print(voxel_coords.shape)
            # all_voexl_gt = voxel_coords[:,[2,3]]
            
            # save_path = '/home/950154_customer/david/OpenPCDet/tmp/'
            # # ground_feature_out_np_save
            # name = str(name_num)+ "_all_label"
            # path = os.path.join(save_path,name+".npy")
            # with open(path, 'wb') as f:
            #     np.save(f, all_voexl_gt)
          
            total_loss = total_loss + loss                                                                                                                                                                  
        
        return total_loss
    # def get_ground_loss_v2(self):
    #     # unet_out = self.forward_ret_dict['unet_out']
    #     # unet_out_features = unet_out.dense()
    #     ground_feature_out = self.forward_ret_dict["ground_feature_out"]
    #     gnd_voxel_coords = self.forward_ret_dict["gnd_voxel_coords"]
    #     # print(type(unet_out_features))
    #     # print(ground_feature_out.size())
    #     # print(gnd_voxel_coords.shape)
    #     # print("---------")
    #     # print(gnd_voxel_coords)
    #     # gnd_voxel_coords = np.delete(gnd_voxel_coords, 0, 2)
    #     # print(gnd_voxel_coords.shape)
    #     # gt_num = gnd_voxel_coords.shape[1]
        
    #     labels = np.zeros([len(gnd_voxel_coords),ground_feature_out.size()[2],ground_feature_out.size()[3]], dtype='float16')
    #     for batch_num,gnd_voxel_coord in enumerate(gnd_voxel_coords):
    #         # z = gnd_voxel_coord[:,0].astype(int)
    #         #  print(gnd_voxel_coord.shape)
    #         # print(gnd_voxel_coord.shape)
    #         y = gnd_voxel_coord[:,1].astype(int)
    #         x = gnd_voxel_coord[:,2].astype(int)
    #         labels[batch_num][y,x] = 1
        
    #     ground_feature_out = ground_feature_out.view(len(gnd_voxel_coords), -1, 2)
    #     labels = torch.from_numpy(labels).cuda().view(len(gnd_voxel_coords), -1)
        
    #     criterion = nn.CrossEntropyLoss()
    #     total_loss = 0

    #     soft_max = nn.Softmax(dim=1)
        
    #     for batch_num in range(len(gnd_voxel_coords)):
            
    #         soft_unet = soft_max(ground_feature_out[batch_num])
    #         soft_unet_max = torch.argmax(soft_unet, dim=1)
    #         train_acc = torch.sum(soft_unet_max == labels[batch_num])
    #         np_soft_unet_max = soft_unet_max.cpu().detach().numpy()
    #         np_labels = labels[batch_num].cpu().detach().numpy()
            
            
            
    #         print("acc = {}".format(float(train_acc)/float(labels[batch_num].size()[0])))
    #         # print("gnd_ratio = {}".format(float(gt_num)/(1024*1024)))
    #         # focal = FocalLoss(gamma = 5)
    #         # loss = focal(unet_out_features[batch_num],labels[batch_num].long())
    #         loss = criterion(ground_feature_out[batch_num], labels[batch_num].long())
    #         total_loss = total_loss + loss
    #     # print(gnd_loss)
    #     return total_loss/len(gnd_voxel_coords) 
    def get_ground_loss(self):
        unet_out = self.forward_ret_dict['unet_out']
        unet_out_features = unet_out.dense()
        # ground_feature_out = self.forward_ret_dict["ground_feature_out"]
        gnd_voxel_coords = self.forward_ret_dict["gnd_voxel_coords"].cpu().detach().numpy()
        # print(type(unet_out_features))
        # print(unet_out_features.shape)
        # print(gnd_voxel_coords.shape)
        # print(gnd_voxel_coords)
        # gnd_voxel_coords = np.delete(gnd_voxel_coords, 0, 2)
        # print(gnd_voxel_coords.shape)
        gt_num = gnd_voxel_coords.shape[1]
        labels = np.zeros([gnd_voxel_coords.shape[0],unet_out_features.size()[2],unet_out_features.size()[3],unet_out_features.size()[4]])
        for batch_num,gnd_voxel_coord in enumerate(gnd_voxel_coords):
            z = gnd_voxel_coord[:,0].astype(int)
            y = gnd_voxel_coord[:,1].astype(int)
            x = gnd_voxel_coord[:,2].astype(int)
            labels[batch_num][z,y,x] = 0
        
        unet_out_features = unet_out_features.view(gnd_voxel_coords.shape[0], -1, 2)
        labels = torch.from_numpy(labels).cuda().view(gnd_voxel_coords.shape[0], -1)
        # print(ground_feature_out.shape)
        # print(labels.shape)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        soft_max = nn.Softmax(dim=1)
        
        for batch_num in range(gnd_voxel_coords.shape[0]):
            soft_unet = soft_max(unet_out_features[batch_num])
            soft_unet_max = torch.argmax(soft_unet, dim=1)
            train_acc = torch.sum(soft_unet_max == labels[batch_num])
            np_soft_unet_max = soft_unet_max.cpu().detach().numpy()
            np_labels = labels[batch_num].cpu().detach().numpy()
            
            # print("np_acc = {}".format(float(len(np.where(np_soft_unet_max==np_labels)[0]))/float(labels[batch_num].size()[0])))
            # print(np.where(np_soft_unet_max==np_labels)[0].shape)
            # print(np.where(np_soft_unet_max==0))
            # print("train_acc = {}".format(train_acc))
            # print("labels[batch_num] = {}".format(labels[batch_num])) 
            # print("soft_unet_max = {}".format(soft_unet_max))    
            print("acc = {}".format(float(train_acc)/float(labels[batch_num].size()[0])))
            print("gnd_ratio = {}".format(float(gt_num)/(1024*1024)))
            # focal = FocalLoss(gamma = 5)
            # loss = focal(unet_out_features[batch_num],labels[batch_num].long())
            loss = criterion(unet_out_features[batch_num], labels[batch_num].long())
            total_loss = total_loss + loss
        # print(gnd_loss)
        return total_loss

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)

        
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]

        # print("cls_preds = {}".format(cls_preds.size()))
        # print("one_hot_targets = {}".format(one_hot_targets.size()))

        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
