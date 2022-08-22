import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import struct
import json
import os
import math
import pcl
class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
        # self.nusc = NuScenes(version='v1.0-trainval', dataroot='/home/950154_customer/david/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)
        self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/maps_with_ground_v2/" # map_with_ground
        # self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/maps_without_ground_v3_hard/" # map_without_ground
    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_nuscenes_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_nuscenes_infos.append(self.infos[k])
            self.infos = sampled_nuscenes_infos
            # self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        # points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 6])[:,[0,1,2,3,5]]
        # print(points)
        #key_rgb = np.concatenate((np.ones([points.shape[0],1])+254,np.zeros([points.shape[0],2])),axis = 1)

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        #num_of_sweep_point = 0
        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

            #num_of_sweep_point = num_of_sweep_point + len(points_sweep)
        #sweep_rgb = np.zeros([num_of_sweep_point,3])+255



        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
        

        # all_rgb = np.concatenate((key_rgb,sweep_rgb),axis = 0)
        # import pptk
        # key_rgb = np.ones([])
        # v = pptk.viewer(points[:,:3],all_rgb)
        # aa = input()


        points = np.concatenate((points, times), axis=1)
        return points
    def quaternion_rotation_matrix(self,Q):
            """
            Covert a quaternion into a full three-dimensional rotation matrix.
            
            Input
            :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)  (w,x,y,z)
            
            Output
            :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                        This rotation matrix converts a point in the local reference 
                        frame to a point in the global reference frame.
            """
            # Extract the values from Q
            q0 = Q[0]
            q1 = Q[1]
            q2 = Q[2]
            q3 = Q[3]
                
            # First row of the rotation matrix
            r00 = 2 * (q0 * q0 + q1 * q1) - 1
            r01 = 2 * (q1 * q2 - q0 * q3)
            r02 = 2 * (q1 * q3 + q0 * q2)
                
            # Second row of the rotation matrix
            r10 = 2 * (q1 * q2 + q0 * q3)
            r11 = 2 * (q0 * q0 + q2 * q2) - 1
            r12 = 2 * (q2 * q3 - q0 * q1)
                
            # Third row of the rotation matrix
            r20 = 2 * (q1 * q3 - q0 * q2)
            r21 = 2 * (q2 * q3 + q0 * q1)
            r22 = 2 * (q0 * q0 + q3 * q3) - 1
                
            # 3x3 rotation matrix
            rot_matrix = np.array([[r00, r01, r02],
                                    [r10, r11, r12],
                                    [r20, r21, r22]])
                                    
            return rot_matrix  
        
    def nuscenes_read_point_cloud(self,path):   
            return np.fromfile(path, dtype=np.float16).reshape(-1,5) # [:,[0,1,2,5]]           
    def read_point_cloud(self,path):   
            return np.fromfile(path, dtype=np.float16).reshape(-1,6) # [:,[0,1,2,5]]
    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        # =========================================================================================
        # nuscenes_data_root = "/home/950154_customer/david/OpenPCDet/data/nuscenes/v1.0-trainval/" 
        ref_from_car = info["ref_from_car"]
        car_from_global = info["car_from_global"]
        sample_token = info["token"]
        
        # sample_record = self.nusc.get('sample', sample_token)
        # scene_token = sample_record['scene_token']
        # scene_record = self.nusc.get('scene', scene_token)
        # log_record = self.nusc.get('log', scene_record['log_token'])
        # map_name = log_record['location']
        # sample_data_token = sample_record['data']['LIDAR_TOP']

        # sample_data_record = self.nusc.get('sample_data', sample_data_token)
        # filename  = sample_data_record["filename"]
        # 
        # 
        # import os
        # # nuscenes_raw_point_cloud = nuscenes_read_point_cloud(os.path.join(nuscenes_data_root,filename))
        # # nuscenes_raw_point_cloud = np.concatenate((nuscenes_raw_point_cloud[:,:3],np.zeros([nuscenes_raw_point_cloud.shape[0],1])),axis = 1)
        # 
        # calibrated_record = self.nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
        # pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        # local_map_translation = pose_record['translation']
        # local_map_translation = np.array(local_map_translation)
        # local_map_rotation =self.quaternion_rotation_matrix(pose_record['rotation'])
        # in_local_map_rotation = np.linalg.inv(local_map_rotation)
        # sensor_local_translation = calibrated_record['translation']
        # sensor_local_translation = np.array(sensor_local_translation)
        # sensor_local_rotation = self.quaternion_rotation_matrix(calibrated_record['rotation'])
        # in_sensor_local_rotation = np.linalg.inv(sensor_local_rotation)
        # sensor_to_local_R = sensor_local_rotation[:3,:3]
        # local_to_map_R = local_map_rotation[:3,:3]

        # map_coor = local_map_translation + sensor_local_translation

        # import json
        # import os
        # import math
        # with open(os.path.join(self.maps_root,map_name,"coordinate_and_scenes_number.json"), 'r') as f:
        #     coordinate_and_scenes_number = json.load(f)

        # center_coordinates = coordinate_and_scenes_number["center_coordinate"]
        # scenes_numbers = coordinate_and_scenes_number["scenes_number"]

        # small_tmp = 9999999
        # small_tmp_array = np.array([9999999,9999998,9999997,9999996])
        # center_coordinate_tmp_array = np.zeros([4,3])
        # scenes_number_tmp_array =  [0,0,0,0]
        # for i,(center_coordinate,scenes_number) in enumerate(zip(center_coordinates,scenes_numbers)):
        #     dis = math.sqrt(np.sum(np.power(np.array(map_coor) - np.array(center_coordinate),2)))
        #     max_idx = np.argmax(small_tmp_array)
        #     if dis - small_tmp_array[max_idx] <0:
        #         small_tmp_array[max_idx] = dis
        #         center_coordinate_tmp_array[max_idx] = center_coordinate
        #         scenes_number_tmp_array[max_idx] = scenes_number
        #     if dis<small_tmp:
        #         small_tmp = dis
        #         scenes_number_tmp = scenes_number
        #         center_x = center_coordinate[0]
        #         center_y = center_coordinate[1]
        #         center_z = center_coordinate[2]
        # 
        # for ii,(center_coordinate,scenes_number_tmp) in enumerate(zip(center_coordinate_tmp_array,scenes_number_tmp_array)):
        #     center_x = center_coordinate[0]
        #     center_y = center_coordinate[1]
        #     center_z = center_coordinate[2]
        #     if map_name=="boston-seaport":
        #         name = "boston_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
        #     if map_name=="singapore-hollandvillage":
        #         name = "hollandvillage_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
        #     if map_name=="singapore-onenorth":
        #         name = "onenorth_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
        #     if map_name=="singapore-queenstown":
        #         name = "queenstown_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
        # 
        # 
        #     
        #     bin_path = os.path.join(self.maps_root,map_name,name)
        #     # map_point_cloud = self.read_point_cloud(bin_path)
        #     map_point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1,6) 
        #     if ii == 0:
        #         all_pcd = map_point_cloud
        #     else:
        #         all_pcd = np.concatenate((map_point_cloud,all_pcd),axis = 0)
        #         # all_pcd = all_pcd + map_point_cloud
        pre_map_path = "/home/950154_customer/david/OpenPCDet/MapDataBin/train/"

        bin_path = os.path.join(pre_map_path,sample_token+".bin")
        map_point_cloud = np.fromfile(bin_path, dtype=np.float16).reshape(-1,5).astype(np.float32)

        # point_cloud_zeros = np.zeros(map_point_cloud[:,:4].shape,dtype = np.float32)
        #             
        # point_cloud_zeros = map_point_cloud[:,:4]
        # p = pcl.PointCloud_PointXYZI(point_cloud_zeros)
        # # 
        # p = pcl.PointCloud_PointXYZI(point_cloud_zeros)
        # sor = p.make_voxel_grid_filter()
        # sor.set_leaf_size(0.2, 0.2, 0.2)
        # cloud_filtered = sor.filter()

        # np_cloud = np.empty([cloud_filtered.width, 4], dtype=np.float32)
        # np_cloud = np.asarray(cloud_filtered)
        # np_cloud = cloud_filtered.to_array()
        # all_pcd = np_cloud
        # all_pcd = map_point_cloud
        
        # print(map_point_cloud.shape) 
        # map_point_cloud_x_y_z = map_point_cloud[:,:3]
        # map_point_cloud_x_y_z = map_point_cloud_x_y_z - local_map_translation
        # map_point_cloud_x_y_z = map_point_cloud_x_y_z.dot(in_local_map_rotation.T)
        # map_point_cloud_x_y_z = map_point_cloud_x_y_z - sensor_local_translation
        # map_point_cloud_x_y_z = map_point_cloud_x_y_z.dot(in_sensor_local_rotation.T)
        # map_point_cloud[:,:3] = map_point_cloud_x_y_z
        all_pcd = map_point_cloud

        all_pcd_x = all_pcd[:,0]
        all_pcd_y = all_pcd[:,1]
        x_idx = np.logical_and(all_pcd_x<=51.2,all_pcd_x>=-51.2)
        y_idx = np.logical_and(all_pcd_y<=51.2,all_pcd_y>=-51.2)
        points_idx = np.logical_and(x_idx,y_idx)
        all_pcd = all_pcd[points_idx] 
        
         
        p = pcl.PointCloud_PointXYZI(all_pcd[:,:4].astype(np.float32))
        # 
        # p = pcl.PointCloud_PointXYZI(point_cloud_zeros)
        sor = p.make_voxel_grid_filter()
        sor.set_leaf_size(0.3, 0.3, 0.3)
        cloud_filtered = sor.filter()

        np_cloud = np.empty([cloud_filtered.width, 4], dtype=np.float32)
        np_cloud = np.asarray(cloud_filtered)
        np_cloud = cloud_filtered.to_array()
        all_pcd = np_cloud

        # all_pcd_x_y_z = np.concatenate((all_pcd[:,:3],np.ones([all_pcd.shape[0],1])),axis = 1)
        # all_pcd_x_y_z = all_pcd_x_y_z.dot(car_from_global.T)
        # all_pcd_x_y_z = all_pcd_x_y_z.dot(ref_from_car.T)
        # all_pcd[:,:3] = all_pcd_x_y_z[:,:3]
        

        # =====================================================================
        # all_pcd_x_y_z = all_pcd[:,:3]
        # all_pcd_x_y_z = all_pcd_x_y_z - local_map_translation
        # all_pcd_x_y_z = all_pcd_x_y_z.dot(in_local_map_rotation.T)
        # all_pcd_x_y_z = all_pcd_x_y_z - sensor_local_translation
        # all_pcd_x_y_z = all_pcd_x_y_z.dot(in_sensor_local_rotation.T)
        # all_pcd[:,:3] = all_pcd_x_y_z
        # ======================================
        all_pcd = np.concatenate((all_pcd[:,:4],np.zeros([all_pcd.shape[0],1])),axis = 1)
        # points = np.concatenate((points[:,:4],np.ones([points.shape[0],1])),axis = 1) 
        all_pcd = np.concatenate((all_pcd,points),axis = 0)        
	
        # points = all_pcd
	# points = all_pcd[:,:4]
        # print(splits.train)
        # print(splits.val)
        # =========================================================================================
        gnd_idx = np.where(map_point_cloud[:,4] == 24)
        # print("----------------------")
        # print(map_point_cloud[:,3])
        # print(map_point_cloud[:,4]) 
        # print(map_point_cloud[:,5])
        # print(len(gnd_idx))
        # print("map_shape = {}".format(map_point_cloud.shape))
        gnd_point_cloud = map_point_cloud[gnd_idx]
        gnd_point_cloud = np.concatenate((gnd_point_cloud[:,:3],np.ones([gnd_point_cloud.shape[0],1],dtype = np.float32)),axis = 1)

        
        p = pcl.PointCloud_PointXYZI(gnd_point_cloud)
        
        # p = pcl.PointCloud_PointXYZI(point_cloud_zeros)
        sor = p.make_voxel_grid_filter()
        sor.set_leaf_size(0.1, 0.1, 0.1)
        gnd_cloud_filtered = sor.filter()

        gnd_np_cloud = np.empty([gnd_cloud_filtered.width, 4], dtype=np.float32)
        gnd_np_cloud = np.asarray(gnd_cloud_filtered)
        gnd_np_cloud = gnd_cloud_filtered.to_array()
        gnd_point_cloud = gnd_np_cloud
        # print('points = {}'.format(points))
        # print('points = {}'.format(points))
        input_dict = {
            'points': all_pcd.astype(np.float16),
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            "gnd_point_cloud": gnd_point_cloud.astype(np.float16)
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None
            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

        del map_point_cloud
        # print(input_dict['gt_boxes'])
        data_dict = self.prepare_data(data_dict=input_dict)
        # print(data_dict['gt_boxes'][:,9])

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
       
        # print(type(data_dict["points"])) 
        if "gnd_point_cloud" in data_dict:
            del data_dict["gnd_point_cloud"]
        if "gnd_voxels" in data_dict:    
            del data_dict["gnd_voxels"]
        if "gnd_voxel_num_points" in data_dict:
            del data_dict["gnd_voxel_num_points"]
        

        # print(data_dict.keys())
        # print(data_dict["gnd_voxel_coords"])
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import nuscenes_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    args = parser.parse_args()

    if args.func == 'create_nuscenes_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )

        nuscenes_dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            logger=common_utils.create_logger(), training=True
        )
        nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
