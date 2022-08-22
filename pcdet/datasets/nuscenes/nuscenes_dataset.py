import open3d as o3d
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
import random



def dir_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir): 
        pass
        # print(len(files)) #當前路徑下所有非目錄子檔案
    files.sort()
    return files

class NuScenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None,map_shift = 0):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_nuscenes_data(self.mode)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='/home/950154_customer/david/OpenPCDet/data/nuscenes/v1.0-trainval', verbose=True)
        # self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/maps_with_ground_v2/" # map_with_ground
        # self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/maps_without_ground_v3_hard/" # map_without_ground 
        # self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/map_by_scenes_v6_Downsampling0.1/" # map_with_ground no intensity

        # self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/map_by_scenes_v7_Downsampling0.1/" # map_with_ground with intensity
        self.maps_root = "/home/950154_customer/david/OpenPCDet/maps/map_by_scenes_v7_Downsampling0.1_no_ground_ieflat" # map_with_ground with intensity with lidar_seg
        
        self.map_shift = map_shift
        
        # ==== 5/28 itri =====
        # # self.itri_lidar_path  = '/home/950154_customer/david/OpenPCDet/20211008-p5-68_2021-10-08-10-42-13_15_bag2bin/lidar'
        # # self.itri_map_path = '/home/950154_customer/david/OpenPCDet/20211008-p5-68_2021-10-08-10-42-13_15_bag2bin/map'

        # # self.itri_lidar_path  = '/home/950154_customer/david/OpenPCDet/20211005-p5-68_2021-10-05-14-27-03_28_bag2bin/lidar_icp'
        # # self.itri_map_path = '/home/950154_customer/david/OpenPCDet/20211005-p5-68_2021-10-05-14-27-03_28_bag2bin/map_icp'

        # # self.itri_lidar_path  = '/home/950154_customer/david/OpenPCDet/20211007-p5-68_2021-10-07-13-41-56_13_bag2bin/lidar_icp'
        # # self.itri_map_path = '/home/950154_customer/david/OpenPCDet/20211007-p5-68_2021-10-07-13-41-56_13_bag2bin/map_icp'

        # # self.itri_lidar_path  = '/home/950154_customer/david/OpenPCDet/20211020-p5-68_2021-10-20-13-39-17_12bag2bin/lidar_icp'
        # # self.itri_map_path = '/home/950154_customer/david/OpenPCDet/20211020-p5-68_2021-10-20-13-39-17_12bag2bin/map_icp'

        # self.itri_lidar_path  = '/home/950154_customer/david/OpenPCDet/20211021-p5-68_2021-10-21-13-32-18_16_bag2bin/lidar_icp'
        # self.itri_map_path = '/home/950154_customer/david/OpenPCDet/20211021-p5-68_2021-10-21-13-32-18_16_bag2bin/map_icp'

        # self.itri_map_name = dir_file_name(self.itri_map_path)
        # ==== 5/28 itri =====

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
        # np.max(points[:,3])
        # print('points = {}'.format(np.max(points[:,3])))
        # print('points = {}'.format(np.max(points[:,4])))
        sample_token = info["token"]

        # ========== 5/28 itri add ===========

        # import os
        # import math
        # theta = 90 * math.pi/180
        # rotation_90_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])

        # each_itri_map_name = self.itri_map_name[index]
        # # print(len(self.itri_map_name))
        # # print(each_itri_map_name)

        # each_itri_map_path = os.path.join(self.itri_map_path,each_itri_map_name)
        # itri_map_points = np.fromfile(each_itri_map_path, dtype=np.float32).reshape(-1,4)
        # itri_map_points[:,:2]  = (rotation_90_matrix.dot(itri_map_points[:,:2].T)).T
        # itri_map_points = np.concatenate((itri_map_points,np.zeros([itri_map_points.shape[0],1])-1),axis = 1)
        # # itri_map_points[:,1] = itri_map_points[:,1] - 51.2
        # # p_filter_idx = itri_map_points[:,1]>51.2
        # # n_filter_idx = itri_map_points[:,1]<-51.2
        # # p_lidar_points = itri_map_points[p_filter_idx] 
        # # p_lidar_points[:,1] = p_lidar_points[:,1] - 51.2
        # # n_lidar_points = itri_map_points[n_filter_idx]
        # # n_lidar_points[:,1] = n_lidar_points[:,1] + 51.2
        # # itri_map_points = np.concatenate((p_lidar_points,n_lidar_points),axis = 0)
        # # print('itri_map_points = {}'.format(itri_map_points))

        # each_itri_lidar_path = os.path.join(self.itri_lidar_path,each_itri_map_name)
        # itri_lidar_points = np.fromfile(each_itri_lidar_path, dtype=np.float32).reshape(-1,4)
        # itri_lidar_points[:,:2]  = (rotation_90_matrix.dot(itri_lidar_points[:,:2].T)).T
        # # print(np.max(itri_lidar_points[:,3]))
        # # print('itri_lidar_points = {}'.format(itri_lidar_points))
        # # itri_lidar_points[:,[3,4]] = itri_lidar_points[:,[4,3]]
        # # itri_lidar_points[:,3] = itri_lidar_points[:,3]*255
        # itri_lidar_points_final = np.concatenate((itri_lidar_points[:,:4],np.zeros([itri_lidar_points.shape[0],1])),axis = 1)
        # # print('itri_lidar_points_final = {}'.format(itri_lidar_points_final))
        # # itri_lidar_points_final = np.concatenate((itri_lidar_points[:,:3],np.zeros([itri_lidar_points.shape[0],1])),axis = 1)
        # # itri_lidar_points_final = np.concatenate((itri_lidar_points_final,itri_lidar_points[:,[3]]),axis = 1)

        # # itri_lidar_points_final[:,1] = itri_lidar_points_final[:,1] -51.2
        # # p_filter_idx = itri_lidar_p`oints_final[:,1]>51.2
        # # n_filter_idx = itri_lidar_points_final[:,1]<-51.2
        # # p_lidar_points = itri_lidar_points_final[p_filter_idx] 
        # # p_lidar_points[:,1] = p_lidar_points[:,1] - 51.2
        # # n_lidar_points = itri_lidar_points_final[n_filter_idx]
        # # n_lidar_points[:,1] = n_lidar_points[:,1] + 51.2
        # # lidar_points = np.concatenate((p_lidar_points,n_lidar_points),axis = 0)

        # # print('itri_lidar_points_final = {}'.format(itri_lidar_points_final))
        # # print('itri_map_points = {}'.format(itri_map_points))
        # # ========== 5/28 itri ===========


# ==============itri 5/28 remove ===================
        # =========================================================================================
        # print(self.map_shift)
        # nuscenes_data_root = "/home/950154_customer/david/OpenPCDet/data/nuscenes/v1.0-trainval/" 
        ref_from_car = info["ref_from_car"]
        car_from_global = info["car_from_global"]
        sample_token = info["token"]
        ## print("sample_token = {}".format(sample_token))
        sample_record = self.nusc.get('sample', sample_token)
        scene_token = sample_record['scene_token']
        scene_record = self.nusc.get('scene', scene_token)
        log_record = self.nusc.get('log', scene_record['log_token'])
        map_name = log_record['location']
        sample_data_token = sample_record['data']['LIDAR_TOP']

        sample_data_record = self.nusc.get('sample_data', sample_data_token)
        filename  = sample_data_record["filename"]
        
        
        import os
        # nuscenes_raw_point_cloud = nuscenes_read_point_cloud(os.path.join(nuscenes_data_root,filename))
        # nuscenes_raw_point_cloud = np.concatenate((nuscenes_raw_point_cloud[:,:3],np.zeros([nuscenes_raw_point_cloud.shape[0],1])),axis = 1)
        
        calibrated_record = self.nusc.get('calibrated_sensor', sample_data_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        local_map_translation = pose_record['translation']
        local_map_translation = np.array(local_map_translation)
        local_map_rotation =self.quaternion_rotation_matrix(pose_record['rotation'])
        in_local_map_rotation = np.linalg.inv(local_map_rotation)
        sensor_local_translation = calibrated_record['translation']
        sensor_local_translation = np.array(sensor_local_translation)
        sensor_local_rotation = self.quaternion_rotation_matrix(calibrated_record['rotation'])
        in_sensor_local_rotation = np.linalg.inv(sensor_local_rotation)
        sensor_to_local_R = sensor_local_rotation[:3,:3]
        local_to_map_R = local_map_rotation[:3,:3]

        map_coor = local_map_translation + sensor_local_translation

        import json
        import os
        import math
        with open(os.path.join(self.maps_root,map_name,"coordinate_and_scenes_number.json"), 'r') as f:
            coordinate_and_scenes_number = json.load(f)

        center_coordinates = coordinate_and_scenes_number["center_coordinate"]
        scenes_numbers = coordinate_and_scenes_number["scenes_number"]

        small_tmp = 9999999
        small_tmp_array = np.array([9999999,9999998,9999997,9999996])
        center_coordinate_tmp_array = np.zeros([4,3])
        scenes_number_tmp_array =  [0,0,0,0]
        for i,(center_coordinate,scenes_number) in enumerate(zip(center_coordinates,scenes_numbers)):

            this_center_coordinate_not_want = False
            for choosed_pt in center_coordinate_tmp_array:

                dist_between_Choosed_ptAndCenter_Coordinate = np.linalg.norm(np.array(choosed_pt)-np.array(center_coordinate))

                if dist_between_Choosed_ptAndCenter_Coordinate < 25:
                    this_center_coordinate_not_want = True
            if this_center_coordinate_not_want:
                continue

            dis = math.sqrt(np.sum(np.power(np.array(map_coor) - np.array(center_coordinate),2)))
            max_idx = np.argmax(small_tmp_array)
            if dis - small_tmp_array[max_idx] <0:
                small_tmp_array[max_idx] = dis
                center_coordinate_tmp_array[max_idx] = center_coordinate
                scenes_number_tmp_array[max_idx] = scenes_number
            if dis<small_tmp:
                small_tmp = dis
                scenes_number_tmp = scenes_number
                center_x = center_coordinate[0]
                center_y = center_coordinate[1]
                center_z = center_coordinate[2]
        
        for ii,(center_coordinate,scenes_number_tmp) in enumerate(zip(center_coordinate_tmp_array,scenes_number_tmp_array)):
            center_x = center_coordinate[0]
            center_y = center_coordinate[1]
            center_z = center_coordinate[2]
            if map_name=="boston-seaport":
                name = "boston_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
            if map_name=="singapore-hollandvillage":
                name = "hollandvillage_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
            if map_name=="singapore-onenorth":
                name = "onenorth_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
            if map_name=="singapore-queenstown":
                name = "queenstown_{}_{}_{}_{}.bin".format(int(round(center_x)),int(round(center_y)),int(round(center_z)),scenes_number_tmp[-4:])
        
        
            
            bin_path = os.path.join(self.maps_root,map_name,name)
            # map_point_cloud = self.read_point_cloud(bin_path)
            # map_point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1,6) # map with ground v2

            map_point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1,5)[:,:4]   # for lidar seg map
            
            if ii == 0:
                all_pcd = map_point_cloud
            else:
                all_pcd = np.concatenate((map_point_cloud,all_pcd),axis = 0)
        # ==================================== itri 5/28 add map ==================================================for gnd map ===============
        # pre_map_path = "/home/950154_customer/david/OpenPCDet/MapDataBin/train/"
        # # pre_map_path = "/home/950154_customer/david/OpenPCDet/data/nuscenes/MapsGT/train/"
        # bin_path = os.path.join(pre_map_path,sample_token+".bin")
        # map_point_cloud = np.fromfile(bin_path, dtype=np.float16).reshape(-1,5).astype(np.float32)


        # map_point_cloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1,4).astype(np.float32)
        # map_point_cloud = np.concatenate((map_point_cloud[:,:4],np.zeros([map_point_cloud.shape[0],1])),axis = 1)
        # points = np.concatenate((points,map_point_cloud),axis = 0)
        # ======================================================================================for gnd map ===============

        # =========================== downsampling ============================
        # rint(map_point_cloud.shape)
        # # print(map_point_cloud)
        # # point_cloud_zeros = np.zeros(map_point_cloud[:,:4].shape,dtype = np.float32)
        # # #             
        # point_cloud_zeros = map_point_cloud[:,:4].astype(np.float32)
        # # p = pcl.PointCloud_PointXYZI(point_cloud_zeros)
        # # 
        # p = pcl.PointCloud_PointXYZI(point_cloud_zeros)
        # sor = p.make_voxel_grid_filter()
        # sor.set_leaf_size(0.1, 0.1, 0.1)
        # cloud_filtered = sor.filter()

        # np_cloud = np.empty([cloud_filtered.width, 4], dtype=np.float32)
        # np_cloud = np.asarray(cloud_filtered)
        # np_cloud = cloud_filtered.to_array()
        # all_pcd = np_cloud
        # # print(point_cloud_zeros.shape)
        # print(np_cloud.shape)
        # # all_pcd = map_point_cloud
        # =========================== down sampling ============================

        # # =====================================================================
        all_pcd_x_y_z = all_pcd[:,:3]
        all_pcd_x_y_z = all_pcd_x_y_z - local_map_translation
        all_pcd_x_y_z = all_pcd_x_y_z.dot(in_local_map_rotation.T)
        all_pcd_x_y_z = all_pcd_x_y_z - sensor_local_translation
        all_pcd_x_y_z = all_pcd_x_y_z.dot(in_sensor_local_rotation.T)
        all_pcd[:,:3] = all_pcd_x_y_z
        # # =====================================

        # =========== 5/28 itri ================
        all_pcd_x = all_pcd[:,0]
        all_pcd_y = all_pcd[:,1]
        x_idx = np.logical_and(all_pcd_x<=51.2,all_pcd_x>=-51.2)
        y_idx = np.logical_and(all_pcd_y<=51.2,all_pcd_y>=-51.2)
        points_idx = np.logical_and(x_idx,y_idx)
        all_pcd = all_pcd[points_idx]
        # # print(all_pcd.shape)
         # =========== 5/28 itri ================


        # ============= test_for _shift ==============
        # theta = random.randint(0,360)
        # pi_angle  = theta * math.pi/180
        # shift_x = math.cos(pi_angle) * self.map_shift
        # shift_y = math.sin(pi_angle) * self.map_shift
        # all_pcd[:,0] = all_pcd[:,0] + shift_x
        # all_pcd[:,1] = all_pcd[:,1] + shift_y
        # # all_pcd[:,1] =  all_pcd[:,1] + self.map_shift
        # ============= test_for _shift ==============
        
        all_pcd = all_pcd[:,:4]
# ==============itri 5/28 remove ===================
        
        # =========== 5/28 itri add===============
        # all_pcd = itri_map_points#[:,:4]
        # points = itri_lidar_points_final
        # =========== 5/28 itri add===============

        # ============ downsampling for gnn === itri 5/28  remove===========
        p = pcl.PointCloud_PointXYZI(all_pcd[:,:4].astype(np.float32)) 
        sor = p.make_voxel_grid_filter()
        sor.set_leaf_size(0.1, 0.1, 0.1)
        cloud_filtered = sor.filter()

        np_cloud = np.empty([cloud_filtered.width, 4], dtype=np.float32)
        np_cloud = np.asarray(cloud_filtered)
        np_cloud = cloud_filtered.to_array()
        all_pcd_for_gnn = np_cloud
        all_pcd_for_gnn = np.concatenate((all_pcd_for_gnn[:,:4],np.zeros([all_pcd_for_gnn.shape[0],1])-1),axis = 1)
        # # ============ downsampling for gnn itri 5/28  remove ==============

        # ============ downsampling for gnn open3d============== 5/28 itri add==========
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(all_pcd[:,:3])
        # # color  = np.concatenate((scene_points_all[:,[3,4]]/255.0,np.zeros([scene_points_all.shape[0],1])),axis = 1)
        # # pcd.colors = o3d.utility.Vector3dVector(color)
        # # pcd.point["intensities"] = o3d.utility.Vector3dVector(all_pcd[:,[3]])
        # # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)
        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # # intensity_load = np.asarray(voxel_down_pcd.colors)
        # # xyz_load = np.asarray(voxel_down_pcd.points)

        # # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=1, radius=0.4)
        
        # # cl = voxel_down_pcd

        # # intensity_load = np.asarray(cl.colors)
        # xyz_load = np.asarray(voxel_down_pcd.points)
        # all_pcd_for_gnn =  np.concatenate((xyz_load[:,:3],np.zeros([xyz_load.shape[0],1])),axis = 1)
        # all_pcd_for_gnn =  np.concatenate((all_pcd_for_gnn[:,:4],np.zeros([all_pcd_for_gnn.shape[0],1])-1),axis = 1)

        # # print("all_pcd_for_gnn = {}".format(all_pcd_for_gnn.shape))
        # # all_pcd_for_gnn = all_pcd
        # ============ downsampling for gnn open3d========5/28 itri add======

        # ============ downsampling for cnn =========5/28 itri add=====
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(all_pcd[:,:3])
        # # color  = np.concatenate((scene_points_all[:,[3,4]]/255.0,np.zeros([scene_points_all.shape[0],1])),axis = 1)
        # # pcd.colors = o3d.utility.Vector3dVector(color)
        # # pcd.point["intensities"] = o3d.utility.Vector3dVector(all_pcd[:,[3]])
        # # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)
        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.4)

        # # intensity_load = np.asarray(voxel_down_pcd.colors)
        # # xyz_load = np.asarray(voxel_down_pcd.points)

        # # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=1, radius=0.4)
        
        # # cl = voxel_down_pcd

        # # intensity_load = np.asarray(cl.colors)
        # xyz_load = np.asarray(voxel_down_pcd.points)
        # all_pcd =  np.concatenate((xyz_load[:,:3],np.zeros([xyz_load.shape[0],1])),axis = 1)
        # all_pcd =  np.concatenate((all_pcd[:,:4],np.zeros([all_pcd.shape[0],1])-1),axis = 1)
        # ============ downsampling for cnn =========5/28 itri add=====

        # ============ downsampling for 128bin =========5/28 itri  add=====
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        # color  = np.concatenate((points[:,[3,4]]/255.0,np.zeros([points.shape[0],1])),axis = 1)
        # pcd.colors = o3d.utility.Vector3dVector(color)
        # # pcd.point["intensities"] = o3d.utility.Vector3dVector(all_pcd[:,[3]])
        # # o3d.io.write_point_cloud("../../TestData/sync.ply", pcd)
        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)

        # # intensity_load = np.asarray(voxel_down_pcd.colors)
        # # xyz_load = np.asarray(voxel_down_pcd.points)

        # # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=1, radius=0.4)
        
        # # cl = voxel_down_pcd

        # intensity_load = np.asarray(voxel_down_pcd.colors)
        # xyz_load = np.asarray(voxel_down_pcd.points)

        # points = np.concatenate((xyz_load[:,:3],intensity_load[:,[0,1]]*255),axis = 1)
        # ============ downsampling for 128bin =========5/28 itri add =====
        


        # # ============ downsampling for cnn ====itri 5/28 remove==========
        p = pcl.PointCloud_PointXYZI(all_pcd[:,:4].astype(np.float32)) 
        sor = p.make_voxel_grid_filter()
        sor.set_leaf_size(0.4, 0.4, 0.4)
        cloud_filtered = sor.filter()

        np_cloud = np.empty([cloud_filtered.width, 4], dtype=np.float32)
        np_cloud = np.asarray(cloud_filtered)
        np_cloud = cloud_filtered.to_array()
        all_pcd = np_cloud
        all_pcd = np.concatenate((all_pcd[:,:4],np.zeros([all_pcd.shape[0],1])-1),axis = 1)
        # # ============ downsampling for cnn remove ==============
        all_pcd = np.concatenate((all_pcd,points),axis = 0)        

        # =========================================================================================

        # ============== save map to bin ===========
        # save_path = "/home/950154_customer/david/OpenPCDet/saved/"
        # name = sample_token + ".bin"
        # path = os.path.join(save_path,name)
        # with open(path, 'wb')as fp:
        #     for x in all_pcd:
        #         for j in x:
        #             a = struct.pack('f', j)
        #             fp.write(a)
        # ============== save map to bin ===========


        input_dict = {
            'points': all_pcd, # points all_pcd
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            "gnd_point_cloud": all_pcd_for_gnn # all_pcd_for_gnn
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

        # del map_point_cloud
        # print(input_dict['gt_boxes'])
        data_dict = self.prepare_data(data_dict=input_dict)
        # print(data_dict['gt_boxes'][:,9])

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False): 
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
       

        # if "gnd_point_cloud" in data_dict:
        #     del data_dict["gnd_point_cloud"]
        # if "gnd_voxels" in data_dict:    
        #     del data_dict["gnd_voxels"]
        # if "gnd_voxel_num_points" in data_dict:
        #     del data_dict["gnd_voxel_num_points"]
        
        # gnd_voxels = data_dict["gnd_voxels"]
        
        
        # print("gnd_voxels = {}".format(gnd_voxels.shape))
        # # print("gnd_voxels = {}".format(gnd_voxels))
        

        # gnd_voxels_labels = gnd_voxels[:,:,5].astype(int)
        
        # gnd_voxels_labels = np.where(gnd_voxels_labels != 0, gnd_voxels_labels, np.nan) 
        
        # gnd_idx = np.where(gnd_voxels_labels == 24)[0]


        # # print(gnd_idx)
        # # u, indices = np.unique(gnd_voxels_labels, return_inverse=True)
        # # voxels_classes = u[np.argmax(np.apply_along_axis(np.bincount, 1, indices.reshape(gnd_voxels_labels.shape),
        # #                                 None, np.max(indices) + 1), axis=1)]
        
        # # # print(np.unique(voxels_classes)+23)

        
        # # data_dict["gnd_voxel_coords"] = data_dict["gnd_voxel_coords"][:,:3]
        # # data_dict["gnd_voxels"] = data_dict["gnd_voxels"][:,:,5]
        # gnd_voxel_coords = data_dict["gnd_voxel_coords"]

        # # print(gnd_voxel_coords.shape)
        # # gnd_voxel_coords = np.concatenate((gnd_voxel_coords,voxels_classes.reshape(voxels_classes.shape[0],1)), axis=1)
        
        # data_dict["gnd_voxel_coords_GT"] = gnd_voxel_coords[gnd_idx]


        # # print(gnd_voxel_coords.shape)

        # # print("989898989898989898989")
        # data_dict["gnd_voxels"] = np.concatenate((gnd_voxels[:,:,[0,1,2,3]],np.zeros([gnd_voxels.shape[0],gnd_voxels.shape[1],1])),axis = 2)
        # print("data_dict[gnd_voxels] = {}".format(data_dict["gnd_voxels"].shape))
        # print("--------------")
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
