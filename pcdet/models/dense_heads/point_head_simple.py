import torch

from ...utils import box_utils
from .point_head_template import PointHeadTemplate


class PointHeadSimple(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.input_channels = input_channels
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )
        
        self.feature_merge_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=self.model_cfg.CLS_FC_feature_out
        )

        self.feature_merge_addmap_residual_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC_ADD_MAP,
            input_channels=192,   # 192   # 320
            output_channels=128   # 128   # 256
        )

        # ========5/8
        # self.feature_merge_addmap_residual_layers_1 = self.make_fc_layers(
        #     fc_cfg=self.model_cfg.CLS_FC_ADD_MAP,
        #     input_channels=320,
        #     output_channels=256
        # )

        # self.feature_merge_addmap_residual_layers_2 = self.make_fc_layers(
        #     fc_cfg=self.model_cfg.CLS_FC_ADD_MAP,
        #     input_channels=320,
        #     output_channels=256
        # )
        # ========5/8

        self.feature_out_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC_out,
            input_channels=128,    # 128 256
            output_channels=num_class
        )
    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']

        # print(gt_boxes)
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        # ===================add======================
        map_pooled_features = batch_dict["map_pooled_features"]
        merged_feature = self.feature_merge_layers(point_features)
        point_features_add_map = torch.cat((merged_feature, map_pooled_features), 1)
        feature_merge_resnet = self.feature_merge_addmap_residual_layers(point_features_add_map) + merged_feature

        # point_features_add_map = torch.cat((point_features, map_pooled_features), 1)
        # feature_merge_resnet = self.feature_merge_addmap_residual_layers(point_features_add_map) + point_features

        # ======5/8
        # point_features_add_map = torch.cat((feature_merge_resnet, map_pooled_features), 1)
        # feature_merge_resnet_2 = self.feature_merge_addmap_residual_layers_1(point_features_add_map) + feature_merge_resnet
        # point_features_add_map = torch.cat((feature_merge_resnet, map_pooled_features), 1)
        # feature_merge_resnet_3 = self.feature_merge_addmap_residual_layers_2(point_features_add_map) + feature_merge_resnet_2
        # feature_merge_resnet = feature_merge_resnet_3
        # ======5/8
        point_cls_preds = feature_merge_resnet
        point_cls_preds = self.feature_out_layers(feature_merge_resnet)
        # ===================add======================

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict
