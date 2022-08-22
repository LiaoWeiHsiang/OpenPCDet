import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points, gnd_point_cloud,map_point_cloud):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        # print(type(gnd_point_cloud))
        if isinstance(gnd_point_cloud,np.ndarray):
            gnd_point_cloud[:, 1] = -gnd_point_cloud[:, 1]
        if isinstance(map_point_cloud,np.ndarray):
            map_point_cloud[:, 1] = -map_point_cloud[:, 1]
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    # flip_T_or_F = enable
    return gt_boxes, points, gnd_point_cloud ,map_point_cloud


def random_flip_along_y(gt_boxes, points, gnd_point_cloud,map_point_cloud):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]
        if isinstance(gnd_point_cloud,np.ndarray):
            gnd_point_cloud[:, 0] = -gnd_point_cloud[:, 0]
        if isinstance(map_point_cloud,np.ndarray):
            map_point_cloud[:, 0] = -map_point_cloud[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points,gnd_point_cloud,gnd_point_cloud


def global_rotation(gt_boxes, points,gnd_point_cloud,map_point_cloud,rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    
    if isinstance(gnd_point_cloud,np.ndarray):
        gnd_point_cloud = common_utils.rotate_points_along_z(gnd_point_cloud[np.newaxis, :, :], np.array([noise_rotation]))[0]
    if isinstance(map_point_cloud,np.ndarray):
        map_point_cloud = common_utils.rotate_points_along_z(map_point_cloud[np.newaxis, :, :], np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    return gt_boxes, points,gnd_point_cloud, map_point_cloud


def global_scaling(gt_boxes, points,gnd_point_cloud,map_point_cloud, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points,gnd_point_cloud,map_point_cloud
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    if isinstance(gnd_point_cloud,np.ndarray):
        gnd_point_cloud[:, :3] *= noise_scale
    if isinstance(map_point_cloud,np.ndarray):
        map_point_cloud[:, :3] *= noise_scale
    
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points , gnd_point_cloud, map_point_cloud
