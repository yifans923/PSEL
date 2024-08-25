import os, sys, glob, h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski
import open3d as o3d

# (9840, 2048, 3), (9840, 1)
# download in：https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
# DATA_DIR = 'data'
# path = os.getcwd()
# p1 = os.path.abspath(os.path.dirname(path)+os.path.sep +".")
DATA_DIR = '/data/syf/data'

Return_un = False

def load_data(partition, file_type='modelnet40'):
    # 读取训练集or测试集
    if file_type == '3DMatch7':
        file_name = '3DMatch/7-scenes-redkitchen/' + partition
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, 'cloud_bin_*.ply')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            # 采样10000个点
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:4096], :]
            all_data.append(points)
        return np.array(all_data), np.array(all_label)
    elif file_type == '3DMatch_all':
        num_points = 4096
        file_name = '3dmatch_all'
        all_data = []
        all_label = []
        DATA_FILES = {
            'train': os.path.join(DATA_DIR, file_name, 'split/train_3dmatch.txt'),  # 1629
            'test': os.path.join(DATA_DIR, file_name, 'split/test_3dmatch.txt'),  # 426
        }
        with open(DATA_FILES[partition], 'r') as f:
            datafile = f.read().split()
        for npz_name in datafile:
            for npz in glob.glob(os.path.join(DATA_DIR, file_name, '{}*.npz'.format(npz_name))):
                f = np.load(npz)
                cur_points = f['pcd'][:].astype(np.float32)
                # cur_colors = f['color'][:].astype(np.float32)
                data = np.concatenate([cur_points], axis=-1).astype(np.float32)
                if data.shape[0] >= num_points:
                    all_data.append(data[:num_points, :])

        return np.array(all_data), np.array(all_label)
    elif file_type == 'Kitti_obj':
        file_name = 'kitti_object/velodyne'
        DATA_FILES = {
            'train': os.path.join(DATA_DIR, file_name + '/training/velodyne'),
            'test': os.path.join(DATA_DIR, file_name + '/testing/velodyne'),
        }
        all_data = []
        all_label = []
        for fname_txt in glob.glob(os.path.join(DATA_FILES[partition], "*.bin")):
            template_data = np.fromfile(fname_txt, dtype=np.float32).reshape(-1, 4)
            points = template_data[:4096, :3]
            # points_idx = np.arange(points.shape[0])
            # np.random.shuffle(points_idx)
            # points = points[points_idx[:2048], :]
            all_data.append(points)
        return np.array(all_data), np.array(all_label)
    elif file_type == 'Kitti_odo':
        file_name = 'KITTI_odometry/sequences/'
        DATA_FILES = {
            'train': ['00', '01'],#, '02', '03', '04', '05'
            'test': ['08'],#, '09', '10'
            'val' : ['06', '07']
        }
        all_data = []
        all_label = []
        all_dataun = []
        for idx in DATA_FILES[partition]:
            for fname in glob.glob(os.path.join(DATA_DIR, file_name, idx, "velodyne", "*.bin")):
                template_data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
                # print(template_data.shape)
                points = template_data[:4096, :3]
                # points_idx = np.arange(points.shape[0])
                # np.random.shuffle(points_idx)
                # points = points[points_idx[:2048], :]
                all_data.append(points)
                all_label.append(0)
                if Return_un:
                    pointsun = template_data[:, :3]
                    all_dataun.append(pointsun)
                    # print(pointsun.shape)
        if Return_un:
            return np.array(all_data), np.array(all_label),np.array(all_dataun)
        return np.array(all_data), np.array(all_label) 
    elif file_type == 'modelnet40':
        file_name = 'modelnet40_ply_hdf5_2048'
    elif file_type == 'S3DIS':
        file_name = 'S3DIS_hdf5'
    elif file_type == 'MVP':
        if partition == 'train':
            file_name = 'mvp_partial/MVP_Train_RG.h5'
            file_name = os.path.join(DATA_DIR, file_name)
            f = h5py.File(file_name)
            data = f['src'][:].astype('float32')
        elif partition == 'test':
            file_name = 'mvp_partial/MVP_Test_RG.h5'
            file_name = os.path.join(DATA_DIR, file_name)
            f = h5py.File(file_name)
            data = f['rotated_src'][:].astype('float32')
        label = f['cat_labels'][:].astype('int64')
        f.close()
        # 取1024个点
        # points_idx = np.arange(data.shape[1])
        # np.random.shuffle(points_idx)
        # data = data[:, points_idx[:1024], :]

        return np.array(data), np.array(label)
    elif file_type == 'Apollo':
        file_name = 'apollo/HighWay/' + partition + '/pcds'
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, '*.pcd')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            # 4096
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:4096], :]
            all_data.append(points)

        return np.array(all_data), np.array(all_label)
    elif file_type == 'bunny':
        file_name = 'bunny/data/'
        all_data = []
        all_label = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, '*.ply')):
            pc = o3d.io.read_point_cloud(h5_name)
            points = normalize_pc(np.array(pc.points))
            # 采样10000个点
            points_idx = np.arange(points.shape[0])
            np.random.shuffle(points_idx)
            points = points[points_idx[:4096], :]
            all_data.append(points)

        return np.array(all_data), np.array(all_label)
    else:
        print('Error file name!')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, file_name, 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        if file_name == 'S3DIS_hdf5':
            data = data[:, :, 0:3]
        label = f['label'][:].astype('int64')
        f.close()
        # 取1024个点
        # points_idx = np.arange(data.shape[1])
        # np.random.shuffle(points_idx)
        # data = data[:, points_idx[:1024], :]
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label  # (9840, 2048, 3), (9840, 1)


def normalize_pc(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(point_cloud) ** 2, axis=-1)))
    point_cloud /= furthest_distance
    return point_cloud


def add_outliers(pointcloud, gt_mask):
    # pointcloud: 			Point Cloud (ndarray) [NxC]
    # output: 				Corrupted Point Cloud (ndarray) [(N+300)xC]
    if isinstance(pointcloud, np.ndarray):
        pointcloud = torch.from_numpy(pointcloud)

    num_outliers = 20
    N, C = pointcloud.shape
    outliers = 2*torch.rand(num_outliers, C)-1 					# Sample points in a cube [-0.5, 0.5]
    pointcloud = torch.cat([pointcloud, outliers], dim=0)
    gt_mask = torch.cat([gt_mask, torch.zeros(num_outliers)])

    idx = torch.randperm(pointcloud.shape[0])
    pointcloud, gt_mask = pointcloud[idx], gt_mask[idx]
    return pointcloud.numpy(), gt_mask


# 加入高斯噪声
def jitter_pointcloud(pointcloud, sigma=0.1, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    # pointcloud += sigma * np.random.randn(N, C)
    return pointcloud


def farthest_subsample_points(pointcloud1, num_subsampled_points):
    # (num_points, 3)
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)

    return pointcloud1[idx1, :], gt_mask

def pc_normalize(pc):
    """
    对点云数据进行归一化
    :param pc: 需要归一化的点云数据
    :return: 归一化后的点云数据
    """
    # 求质心，也就是一个平移量，实际上就是求均值
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # 对点云进行缩放
    pc = pc / m
    return pc


class ModelNet40_Reg(Dataset):
    def __init__(self, num_subsampled_rate, partition='train', max_angle=45, max_t=0.5,
                 noise=False, partial_source=False, unseen=False, file_type='modelnet40'):
        self.partial_source = partial_source  # 是否部分重叠(第二个点云部分缺失)
        if Return_un:
            self.data, self.label, self.dataun = load_data(partition, file_type=file_type)
        else:
            self.data, self.label = load_data(partition, file_type=file_type)
        print('data load finish!')
        self.file_type = file_type
        self.partition = partition
        self.label = self.label.squeeze()  # 去掉维度为1的条目
        self.max_angle = np.pi / 180 * max_angle
        self.max_t = max_t
        self.noise = noise
        self.unseen =unseen
        self.num_subsampled_rate = num_subsampled_rate

        if file_type == 'modelnet40' and self.unseen:
            # simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]
        # elif file_type == 'Kitti_odo':
        #     self.data = pc_normalize(self.data)

    def __getitem__(self, item):
        if self.file_type == 'modelnet40':
            pointcloud = self.data[item][:1024]
        else:
            pointcloud = self.data[item]
            if Return_un:
                pointcloudun = self.dataun[item]
        # pointcloud = self.data[item]
        # pointcloud = jitter_pointcloud(pointcloud)
        if self.max_angle == 45:
            anglex = np.random.uniform(-self.max_angle, self.max_angle)
            angley = np.random.uniform(-self.max_angle, self.max_angle)
            anglez = np.random.uniform(-self.max_angle, self.max_angle)
        else:
            anglex = np.random.uniform(0, self.max_angle)
            angley = np.random.uniform(0, self.max_angle)
            anglez = np.random.uniform(0, self.max_angle)
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        euler_ab = np.asarray([anglez, angley, anglex])
        # 平移矩阵T
        translation_ab = np.array([np.random.uniform(-self.max_t, self.max_t), np.random.uniform(-self.max_t, self.max_t),
                                   np.random.uniform(-self.max_t, self.max_t)])
        # translation_ba = -R_ba.dot(translation_ab)
        # 第item个物体 点云1 [3xN]
        pointcloud1 = pointcloud.T
        # euler_ba = -euler_ab[::-1]
        # 打乱点的顺序(3, num_points)
        pointcloud1 = np.random.permutation(pointcloud1.T).T
        
        if Return_un:
            pointcloud1un = pointcloudun.T 

        # 是否部分重叠
        if self.partial_source:
            # (num_points, 3)
            num_subsampled_points = int(self.num_subsampled_rate * pointcloud1.shape[1])
            pointcloud2, gt_mask = farthest_subsample_points(pointcloud1.T, num_subsampled_points)
            if Return_un:
                num_subsampled_points = int(self.num_subsampled_rate * pointcloud1un.shape[1])
                pointcloud2un, _ = farthest_subsample_points(pointcloud1un.T, num_subsampled_points)
                pointcloud2un = rotation_ab.apply(pointcloud2un).T + np.expand_dims(translation_ab, axis=1)
            # (3, num_points)
            pointcloud2 = rotation_ab.apply(pointcloud2).T + np.expand_dims(translation_ab, axis=1)
            if self.noise:
                # ---加入噪声---
                # (num_points, 3)
                pointcloud2 = jitter_pointcloud(pointcloud2.T,sigma=0.01)
                pointcloud2 = pointcloud2.T

                # pointcloud1, gt_mask = add_outliers(pointcloud1.T, gt_mask)
                # pointcloud1 = pointcloud1.T  # (3, num_points)
                # ---end---
            if Return_un:
                return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
                   translation_ab.astype('float32'), euler_ab.astype('float32'), gt_mask ,pointcloud1un.astype('float32'),pointcloud2un.astype('float32')
            return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
                   translation_ab.astype('float32'), euler_ab.astype('float32'), gt_mask 

        else:
            # 将点云1按角度旋转
            pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)
            pointcloud2 = np.random.permutation(pointcloud2.T).T
            gt_mask = torch.tensor([0,0,0])

            # return (batch, 3, num_points)
            # 两个点云，旋转矩阵R_ab，T_ab, 欧拉角，点云1旋转平移得到点云2
            return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
                   translation_ab.astype('float32'), euler_ab.astype('float32'), gt_mask

    def __len__(self):
        return self.data.shape[0]



