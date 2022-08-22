import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        # gnd_voxel_features, gnd_voxel_num_points = batch_dict['gnd_voxels'], batch_dict['gnd_voxel_num_points']
        # # gnd_voxel_features = gnd_voxel_features
        # gnd_points_mean = gnd_voxel_features[:, :, :].sum(dim=1, keepdim=False)
        # gnd_normalizer = torch.clamp_min(gnd_voxel_num_points.view(-1, 1), min=1.0).type_as(gnd_voxel_features)
        # # print(gnd_points_mean.size())
        # # print(gnd_normalizer.size())
        # gnd_points_mean = gnd_points_mean / gnd_normalizer
        # batch_dict['gnd_voxel_features'] = gnd_points_mean.contiguous()
        return batch_dict
