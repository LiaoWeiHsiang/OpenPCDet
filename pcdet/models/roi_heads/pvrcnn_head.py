import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from .gnn import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d

def multi_layer_downsampling(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False,):
    """Downsample the points using base_voxel_size at different scales"""
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    downsampled_list = [points_xyz]
    last_level = 0
    for level in levels:
        if np.isclose(last_level, level):
            downsampled_list.append(np.copy(downsampled_list[-1]))
        else:
            if add_rnd3d:
                xyz_idx = (points_xyz-xyz_offset+
                    base_voxel_size*level*np.random.random((1,3)))//\
                        (base_voxel_size*level)
                xyz_idx = xyz_idx.astype(np.int32)
                dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
                keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+\
                    xyz_idx[:, 2]*dim_y*dim_x
                sorted_order = np.argsort(keys)
                sorted_keys = keys[sorted_order]
                sorted_points_xyz = points_xyz[sorted_order]
                _, lens = np.unique(sorted_keys, return_counts=True)
                indices = np.hstack([[0], lens[:-1]]).cumsum()
                downsampled_xyz = np.add.reduceat(
                    sorted_points_xyz, indices, axis=0)/lens[:,np.newaxis]
                downsampled_list.append(np.array(downsampled_xyz))
            else:
                
                pcd = open3d.PointCloud()
                pcd.points = open3d.Vector3dVector(points_xyz)
                downsampled_xyz = np.asarray(open3d.voxel_down_sample(
                    pcd, voxel_size = base_voxel_size*level).points)
                downsampled_list.append(downsampled_xyz)
        last_level = level
    return downsampled_list

def multi_layer_downsampling_select(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales and match the downsampled
    points to original points by a nearest neighbor search.
    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
    returns: vertex_coord_list, keypoint_indices_list
    """
    # Voxel downsampling
    vertex_coord_list = multi_layer_downsampling(
        points_xyz, base_voxel_size, levels=levels, add_rnd3d=add_rnd3d)
    num_levels = len(vertex_coord_list)
    assert num_levels == len(levels) + 1
    # Match downsampled vertices to original by a nearest neighbor search.
    keypoint_indices_list = []
    last_level = 0
    for i in range(1, num_levels):
        current_level = levels[i-1]
        base_points = vertex_coord_list[i-1]
        current_points = vertex_coord_list[i]
        if np.isclose(current_level, last_level):
            # same downsample scale (gnn layer),
            # just copy it, no need to search.
            vertex_coord_list[i] = base_points
            keypoint_indices_list.append(
                np.expand_dims(np.arange(base_points.shape[0]),axis=1))
        else:
            # different scale (pooling layer), search original points.
            nbrs = NearestNeighbors(n_neighbors=1,
                algorithm='kd_tree', n_jobs=1).fit(base_points)
            indices = nbrs.kneighbors(current_points, return_distance=False)
            vertex_coord_list[i] = base_points[indices[:, 0], :]
            keypoint_indices_list.append(indices)
        last_level = current_level
    return vertex_coord_list, keypoint_indices_list

def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales by randomly select a point
    within a voxel cell.
    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
    returns: vertex_coord_list, keypoint_indices_list
    """
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    vertex_coord_list = [points_xyz]
    keypoint_indices_list = []
    last_level = 0
    for level in levels:
        last_points_xyz = vertex_coord_list[-1]
        if np.isclose(last_level, level):
            # same downsample scale (gnn layer), just copy it
            vertex_coord_list.append(np.copy(last_points_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
        else:
            if not add_rnd3d:
                xyz_idx = (last_points_xyz - xyz_offset) \
                    // (base_voxel_size*level)
            else:
                xyz_idx = (last_points_xyz - xyz_offset +
                    base_voxel_size*level*np.random.random((1,3))) \
                        // (base_voxel_size*level)
            xyz_idx = xyz_idx.astype(np.int32)
            dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
            keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+xyz_idx[:, 2]*dim_y*dim_x
            num_points = xyz_idx.shape[0]

            voxels_idx = {}
            for pidx in range(len(last_points_xyz)):
                key = keys[pidx]
                if key in voxels_idx:
                    voxels_idx[key].append(pidx)
                else:
                    voxels_idx[key] = [pidx]

            downsampled_xyz = []
            downsampled_xyz_idx = []
            for key in voxels_idx:
                center_idx = random.choice(voxels_idx[key])
                downsampled_xyz.append(last_points_xyz[center_idx])
                downsampled_xyz_idx.append(center_idx)
            vertex_coord_list.append(np.array(downsampled_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.array(downsampled_xyz_idx),axis=1))
        last_level = level

    return vertex_coord_list, keypoint_indices_list
def gen_disjointed_rnn_local_graph_v3(
    points_xyz, center_xyz, radius, num_neighbors,
    neighbors_downsample_method='random',
    scale=None):
    """Generate a local graph by radius neighbors.
    """
    if scale is not None:
        scale = np.array(scale)
        points_xyz = points_xyz/scale
        center_xyz = center_xyz/scale
    nbrs = NearestNeighbors(
        radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
    indices = nbrs.radius_neighbors(center_xyz, return_distance=False)
    if num_neighbors > 0:
        if neighbors_downsample_method == 'random':
            indices = [neighbors if neighbors.size <= num_neighbors else
                np.random.choice(neighbors, num_neighbors, replace=False)
                for neighbors in indices]
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate(
        [i*np.ones(neighbors.size, dtype=np.int32)
            for i, neighbors in enumerate(indices)])
    vertices = np.array([vertices_v, vertices_i]).transpose()
    return vertices
def get_graph_generate_fn(method_name):
    method_map = {
        'disjointed_rnn_local_graph_v3':gen_disjointed_rnn_local_graph_v3,
        'multi_level_local_graph_v3': gen_multi_level_local_graph_v3,
    }
    return method_map[method_name]
def gen_multi_level_local_graph_v3(
    points_xyz, base_voxel_size, level_configs, add_rnd3d=False,
    downsample_method='center'):
    """Generating graphs at multiple scale. This function enforce output
    vertices of a graph matches the input vertices of next graph so that
    gnn layers can be applied sequentially.
    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
        downsample_method: string, the name of downsampling method.
    returns: vertex_coord_list, keypoint_indices_list, edges_list
    """
    if isinstance(base_voxel_size, list):
        base_voxel_size = np.array(base_voxel_size)
    # Gather the downsample scale for each graph
    scales = [config['graph_scale'] for config in level_configs]
    # Generate vertex coordinates
    if downsample_method=='center':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_select(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    if downsample_method=='random':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_random(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    # Create edges
    edges_list = []
    for config in level_configs:
        graph_level = config['graph_level']
        gen_graph_fn = get_graph_generate_fn(config['graph_gen_method'])
        method_kwarg = config['graph_gen_kwargs']
        points_xyz = vertex_coord_list[graph_level]
        center_xyz = vertex_coord_list[graph_level+1]
        vertices = gen_graph_fn(points_xyz, center_xyz, **method_kwarg)
        edges_list.append(vertices)
    return vertex_coord_list, keypoint_indices_list, edges_list

class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )
        self.roi_grid_pool_layer_reg = pointnet2_stack_modules.StackSAModuleMSG_for_reg(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )
        
        # self.roi_grid_pool_layer_for_cls = pointnet2_stack_modules.StackSAModuleMSG_for_cls(
        #     radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
        #     nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
        #     mlps=mlps,
        #     use_xyz=True,
        #     pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        # )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)
        # ================ add for divide cls and reg =============
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list_reg = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list_reg.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list_reg.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer_reg = nn.Sequential(*shared_fc_list_reg)
        # ================ add for divide cls and reg =============
        # self.shared_fc_layer_for_cls = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')


        self.point_set_pooling = PointSetPooling()

        self.graph_nets = nn.ModuleList()
        for i in range(3):
            self.graph_nets.append(GraphNetAutoCenter())

        self.point_set_pooling_out = PointSetPooling_out()  

        self.merge_two_feature = multi_layer_neural_network_fn([512,256,128,64,32,self.num_class])

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        # print('roi_shape = {}'.format(rois.shape))
        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features
    def roi_grid_pool_reg(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features_not_add_map']
        # print(batch_dict['point_cls_scores'].shape)
        # point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer_reg(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features
    # def roi_grid_pool_for_cls(self, batch_dict):
    #     """
    #     Args:
    #         batch_dict:
    #             batch_size:
    #             rois: (B, num_rois, 7 + C)
    #             point_coords: (num_points, 4)  [bs_idx, x, y, z]
    #             point_features: (num_points, C)
    #             point_cls_scores: (N1 + N2 + N3 + ..., 1)
    #             point_part_offset: (N1 + N2 + N3 + ..., 3)
    #     Returns:

    #     """
    #     batch_size = batch_dict['batch_size']
    #     rois = batch_dict['rois']
    #     point_coords = batch_dict['point_coords']
    #     point_features = batch_dict['point_features_for_GNN_cls']

    #     point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

    #     global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
    #         rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
    #     )  # (BxN, 6x6x6, 3)
    #     global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

    #     xyz = point_coords[:, 1:4]
    #     xyz_batch_cnt = xyz.new_zeros(batch_size).int()
    #     batch_idx = point_coords[:, 0]
    #     for k in range(batch_size):
    #         xyz_batch_cnt[k] = (batch_idx == k).sum()

    #     new_xyz = global_roi_grid_points.view(-1, 3)
    #     new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])

    #     pooled_points, pooled_features = self.roi_grid_pool_layer_for_cls(
    #         xyz=xyz.contiguous(),
    #         xyz_batch_cnt=xyz_batch_cnt,
    #         new_xyz=new_xyz,
    #         new_xyz_batch_cnt=new_xyz_batch_cnt,
    #         features=point_features.contiguous(),
    #     )  # (M1 + M2 ..., C)

    #     pooled_features = pooled_features.view(
    #         -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
    #         pooled_features.shape[-1]
    #     )  # (BxN, 6x6x6, C)
    #     return pooled_features
    

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        
        #-------------------------------------------------------
       # import pickle
       # print(batch_dict['frame_id'])
       # # print(batch_dict.keys())
       # # if batch_dict['frame_id'][0] == 'n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd':  # this is for trainval dataset data
       # if batch_dict['frame_id'][0] == 'n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd': # this is for mini dataset validation data
       #     all_proposal = []
       # else :
       #     with open("/home/950154_customer/david/OpenPCDet/proposal.pkl", "rb") as input_file:
       #         all_proposal = pickle.load(input_file)
       # proposal = {}
       # output_roi = batch_dict['rois'].cpu().detach().numpy().squeeze()
       # tmp_dict = {'output_roi':output_roi}
       # proposal.update(tmp_dict)
       # output_roi_scores = batch_dict['roi_scores'].cpu().detach().numpy().squeeze()
       # tmp_dict = {'output_roi_scores':output_roi_scores}
       # proposal.update(tmp_dict)
       # output_roi_labels = batch_dict['roi_labels'].cpu().detach().numpy().squeeze()
       # tmp_dict = {'output_roi_labels':output_roi_labels}
       # proposal.update(tmp_dict)
       # output_frame_id = batch_dict['frame_id']# .cpu().detach().numpy().squeeze()
       # tmp_dict = {'output_frame_id':output_frame_id}
       # proposal.update(tmp_dict)
       # output_roi_metadata = batch_dict['metadata'].cpu().detach().numpy().squeeze()                                                                                                                                              
       # tmp_dict = {'output_roi_metadata':output_roi_metadata}
       # # batch_dict['metadata']
       # proposal.update(tmp_dict)

       # all_proposal = all_proposal + [proposal]
       # # print('proposal = {}'.format(proposal))

       # with open("/home/950154_customer/david/OpenPCDet/proposal.pkl", "wb") as output_file:
       #     pickle.dump(all_proposal, output_file)
        #------------------------------------------------------ 
        # RoI aware pooling

        
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        

        # ======================= add for divide cls and reg ===================

        pooled_features_reg = self.roi_grid_pool_reg(batch_dict)
        # print("pooled_features_reg = {}".format(pooled_features_reg.shape))
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn_reg = pooled_features_reg.shape[0]
        pooled_features_reg = pooled_features_reg.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn_reg, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        
        
        # print("pooled_features_reg = {}".format(pooled_features_reg.shape))
        shared_features_reg = self.shared_fc_layer_reg(pooled_features_reg.view(batch_size_rcnn_reg, -1, 1))
        # ======================= add for divide cls and reg ===================

        
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        


        # =================== for cls ================
        # pooled_features = self.roi_grid_pool_for_cls(batch_dict)  # (BxN, 6x6x6, C)

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features_for_cls = self.shared_fc_layer_for_cls(pooled_features.view(batch_size_rcnn, -1, 1))

  

        
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)

        # print(rcnn_cls.shape)
        # ======== modify for divide cls and reg =========
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        rcnn_reg = self.reg_layers(shared_features_reg).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        # ======== modify for divide cls and reg =========
        
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        # print("=========2")
        # print(rcnn_cls.size())
        # print("=========2")

        


    #  ===================== add graph maps =======================
        # batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
        #         batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
        #     )
        # # print("batch_box_preds_shape = {}".format(batch_box_preds.shape))
        
        # # batch_box_preds = batch_dict["batch_box_preds"].cpu().detach().numpy()
        # # batch_cls_preds = batch_dict["batch_cls_preds"].cpu().detach().numpy()
        # batch_box_preds = batch_box_preds.cpu().detach().numpy()
        # batch_cls_preds = batch_cls_preds.cpu().detach().numpy()
        
        # # print("batch_box_preds_shape = {}".format(batch_box_preds.shape))
        # batch_points = batch_dict["points"].cpu().detach().numpy()

        # batch_map_points = batch_dict["gnd_point_cloud"].cpu().detach().numpy()

        # print(batch_map_points.shape)

        # batch_size = len(batch_box_preds)
        # # print(batch_map_points)
        # import numpy as np
        # for batch_num in range(batch_size):
        #     # points = batch_points[np.where(batch_points[:,0]==batch_num)[0]][:,[1,2,3,4]]
        #     map_points = batch_map_points[np.where(batch_map_points[:,0]==batch_num)[0]][:,[1,2,3,4]]
        #     box_preds = batch_box_preds[batch_num]
        #     cls_preds = batch_cls_preds[batch_num]

        #     # # print(batch_num)

        #     # # print(len(box_preds))

        #     # # indice = []
        #     # for k in range(len(box_preds)):
        #     #     # print(rcnn_cls[k])


        #     #     dis = np.linalg.norm(points[:,:3] - box_preds[k,:3],axis = 1)       
        #     #     idx = np.where(dis<=6)
        #     #     # inside_radi_points = map_points[idx[0]]
        #     #     # print(idx)
        #     #     if k == 0:
        #     #         # all_pcd = inside_radi_points
        #     #         indice = idx[0]
        #     #     else:
        #     #         # all_pcd = np.concatenate((all_pcd,inside_radi_points),axis = 0)
        #     #         indice = np.concatenate((indice,idx[0]),axis = 0)
        
        #     #     # print(inside_radi_points.shape)
        #     # # inside_radi_points = inside_radi_points.reshape(inside_radi_points.shape[0],inside_radi_points.shape[1],1)
        #     # indice = np.unique(indice)
        #     # points = points[indice]
        #     # # inside_radi_points = np.squeeze(inside_radi_points, axis=2)

        # level_configs = [
        #                     {
        #                         "graph_gen_kwargs": {
        #                             "num_neighbors": -1,
        #                             "radius": 1
        #                         },
        #                         "graph_gen_method": "disjointed_rnn_local_graph_v3",
        #                         "graph_level": 0,
        #                         "graph_scale": 1
        #                     },
        #                     {
        #                         "graph_gen_kwargs": {
        #                             "num_neighbors": 256,
        #                             "radius": 4.0
        #                         },
        #                         "graph_gen_method": "disjointed_rnn_local_graph_v3",
        #                         "graph_level": 1,
        #                         "graph_scale": 1
        #                     }
        #                     ]
        # vertex_coord_list, keypoint_indices_list, edges_list = gen_multi_level_local_graph_v3(map_points[:,:3],2,level_configs)

        # point_features, point_coordinates, keypoint_indices, set_indices = map_points[:,[3]], vertex_coord_list[0], keypoint_indices_list[0], edges_list[0]

        # # print(point_features.shape)
        # # print(point_coordinates.shape)
        # point_features = torch.from_numpy(point_features).cuda()
        # point_coordinates = torch.from_numpy(point_coordinates).cuda()
        # keypoint_indices = torch.from_numpy(keypoint_indices).cuda()
        # set_indices = torch.from_numpy(set_indices).cuda()

        # point_features = self.point_set_pooling(point_features, point_coordinates, keypoint_indices, set_indices)
        
        # point_coordinates, keypoint_indices, set_indices = vertex_coord_list[1], keypoint_indices_list[1], edges_list[1]
        

        # point_coordinates = torch.from_numpy(point_coordinates).cuda()
        # keypoint_indices = torch.from_numpy(keypoint_indices).cuda()
        # set_indices = torch.from_numpy(set_indices).cuda()
        # for i, graph_net in enumerate(self.graph_nets):
        #     point_features = graph_net(point_features, point_coordinates, keypoint_indices, set_indices)


        # keypoints_indice = []
        

        # for k in range(len(box_preds)):
        #         # print(rcnn_cls[k])


        #         dis = np.linalg.norm(point_coordinates.cpu().detach().numpy()[:,:3] - box_preds[k,:3],axis = 1)       
        #         idx = np.where(dis<=6)
        #         # inside_radi_points = map_points[idx[0]]
        #         # print(idx)
        #         # if k == 0:
        #         #     # all_pcd = inside_radi_points
        #         #     indice = idx[0]
        #         # else:
        #         #     # all_pcd = np.concatenate((all_pcd,inside_radi_points),axis = 0)
        #         #     indice = np.concatenate((indice,idx[0]),axis = 0)
        #         # print(point_features[idx[0]].size())

        #         if k == 0:
        #             keypoint_feature = point_features[idx[0]]
        #         else:
        #             keypoint_feature = torch.cat((keypoint_feature,point_features[idx[0]]),0)

                
        #         for jj in range(point_features[idx[0]].size()[0]):
        #             keypoints_indice.append(k)
        
        # keypoints_indice = torch.LongTensor(keypoints_indice).cuda()
        # out = self.point_set_pooling_out(keypoint_feature,len(box_preds),keypoints_indice)

        # shared_features_squeeze = torch.squeeze(shared_features)
        # merge_feature = torch.cat((shared_features_squeeze,out),1)

        
        # after_merge = self.merge_two_feature(merge_feature)
        # print(out.size())
        # print(after_merge.size())
        #  ===================== add graph maps =======================


        # # ========================== save point & bbox ==========================
        # # print("batch_cls_preds_shape = {}".format(batch_cls_preds.shape))
        # import pickle
        # batch_dict_path = "/home/950154_customer/david/OpenPCDet/saved/batch_dict"
        # save_dict = {}
        # save_dict = {# "class":batch_cls_preds,\
        # # "box":batch_box_preds,\
        # # "map_points":map_points,\
        # # "key_points":vertex_coord_list[1],\
        # "points":batch_points}
        # with open(batch_dict_path + '.pkl', 'wb') as f:
        #     pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)








        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )


            # print(batch_dict['cls_preds_for_heatmap'])

            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        # print(batch_cls_preds.shape)
        # print(batch_box_preds.shape)

        return batch_dict
