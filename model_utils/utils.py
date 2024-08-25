#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
import sys, os
sys.path.append("..")
import torch.nn.functional as F
from torch.autograd import Variable
from chamfer_loss import *

_EPS = 1e-5
import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from matplotlib import cm
from open3d.visualization import draw_geometries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
def vis_pc_color(pc,score):
    # scores: list [(N,)]

    # score = np.sum(score, axis=1)  # src正号，tgt负号
    # score = np.max(score, axis=1)  # src正号，tgt负号
    # score = score[:,2] # src正号，tgt负号
    score = np.mean((score)**4, axis=1)
    hlist = list(score.flatten())
    count = [0,0,0,0,0,0,0,0,0,0]
    for i in range(10):
        for num in hlist:
            if i/10.0 <= num < (i+1)/10.0:
                count[i]+=1
        print(count[i])
    # plt.bar(range(len(count)), count)
    
    data = pd.DataFrame({'x': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,0.9,1], 'height': count, 'color_list': ['r', 'g', 'b', 'y', 'c', 'm', 'k','r', 'g', 'b']})

    data.plot.bar(x='x', y='height', color=data['color_list'], legend=False)
    
    plt.show()
    # score = np.mean((score)*0, axis=1)
    # score = np.zeros(score.shape,dtype=int)
    pcs = []
    color = get_color_map(score)
    
    pcd = make_open3d_point_cloud(pc, color)
    pcs.append(pcd)

    draw_geometries(pcs)
    
def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd
    
def get_color_map(x):
    # viridis = cm.get_cmap('spring', 512)
    # viridis = cm.get_cmap('cool', 64)
    # viridis = cm.get_cmap('Greens', 512)
    viridis = cm.get_cmap('rainbow', 32) 
    colours = viridis(x).squeeze()
    return colours[:, :3]
def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

def batch_frobenius_norm(matrix1, matrix2):
    loss_F = torch.norm((matrix1 - matrix2), dim=(1, 2))
    return loss_F

def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return: 
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(2,1))
    tr=np.trace(R,0,1,2) 
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs

def square_dists(points1, points2):
    '''
    Calculate square dists between two group points
    :param points1: shape=(B, N, C)
    :param points2: shape=(B, M, C)
    :return:
    '''
    B, N, C = points1.shape
    _, M, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, N, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, M)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    #dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return dists.float()
def ball_query(xyz, new_xyz, radius, K, rt_density=False):
    '''

    :param xyz: shape=(B, N, 3)
    :param new_xyz: shape=(B, M, 3)
    :param radius: int
    :param K: int, an upper limit samples
    :return: shape=(B, M, K)
    '''
    device = xyz.device
    B, N, C = xyz.shape
    M = new_xyz.shape[1]
    grouped_inds = torch.arange(0, N, dtype=torch.long).to(device).view(1, 1, N).repeat(B, M, 1)
    dists = square_dists(new_xyz, xyz)
    grouped_inds[dists > radius ** 2] = N
    if rt_density:
        density = torch.sum(grouped_inds < N, dim=-1)
        density = density / N
    grouped_inds = torch.sort(grouped_inds, dim=-1)[0][:, :, :K]
    grouped_min_inds = grouped_inds[:, :, 0:1].repeat(1, 1, K)
    grouped_inds[grouped_inds == N] = grouped_min_inds[grouped_inds == N]
    if rt_density:
        return grouped_inds, density
    return grouped_inds


def sample_and_group(xyz, points, M, radius, K, use_xyz=True, rt_density=False):
    '''
    :param xyz: shape=(B, N, 3)
    :param points: shape=(B, N, C)
    :param M: int
    :param radius:float
    :param K: int
    :param use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    :return: new_xyz, shape=(B, M, 3); new_points, shape=(B, M, K, C+3);
             group_inds, shape=(B, M, K); grouped_xyz, shape=(B, M, K, 3)
    '''
    if M < 0:
        new_xyz = xyz
    else:
        new_xyz = gather_points(xyz, fps(xyz, M))
    if rt_density:
        grouped_inds, density = ball_query(xyz, new_xyz, radius, K,
                                           rt_density=True)
    else:
        grouped_inds = ball_query(xyz, new_xyz, radius, K, rt_density=False)
    grouped_xyz = gather_points(xyz, grouped_inds)
    grouped_xyz -= torch.unsqueeze(new_xyz, 2).repeat(1, 1, K, 1)
    if points is not None:
        grouped_points = gather_points(points, grouped_inds)
        if use_xyz:
            new_points = torch.cat((grouped_xyz.float(), grouped_points.float()), dim=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    if rt_density:
        return new_xyz, new_points, grouped_inds, grouped_xyz, density
    return new_xyz, new_points, grouped_inds, grouped_xyz


class LocalFeatureFused(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(LocalFeatureFused, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.blocks.add_module(f'conv2d_{i}',
                                   nn.Conv2d(in_dim, out_dim, 1, bias=False))
            self.blocks.add_module(f'gn_{i}',
                                   nn.GroupNorm(out_dim // 32, out_dim))
            self.blocks.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        '''

        :param x: (B, C1, K, M)
        :return: (B, C2, M)
        '''
        x = self.blocks(x)
        x = torch.max(x, dim=2)[0]
        return x


class LocalFeatue(nn.Module):
    def __init__(self, radius, K, in_dim, out_dims):
        super(LocalFeatue, self).__init__()
        self.radius = radius
        self.K = K
        self.local_feature_fused = LocalFeatureFused(in_dim=in_dim,
                                                     out_dims=out_dims)

    def forward(self, feature, xyz, permute=False, use_ppf=False):
        '''
        :param feature: (B, N, C1) or (B, C1, N) for permute
        :param xyz: (B, N, 3)
        :return: (B, C2, N)
        '''
        if permute:
            feature = feature.permute(0, 2, 1).contiguous()
        new_xyz, new_points, grouped_inds, grouped_xyz = \
            sample_and_group(xyz=xyz,
                             points=feature,
                             M=-1,
                             radius=self.radius,
                             K=self.K)
        if use_ppf:
            nr_d = angle(feature[:, :, None, :], grouped_xyz)
            ni_d = angle(new_points[..., 3:], grouped_xyz)
            nr_ni = angle(feature[:, :, None, :], new_points[..., 3:])
            d_norm = torch.norm(grouped_xyz, dim=-1)
            ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1) # (B, N, K, 4)
            new_points = torch.cat([new_points[..., :3], ppf_feat], dim=-1)
        xyz = torch.unsqueeze(xyz, dim=2).repeat(1, 1, self.K, 1)
        new_points = torch.cat([xyz, new_points], dim=-1)
        feature_local = new_points.permute(0, 3, 2, 1).contiguous() # (B, C1 + 3, K, M)
        feature_local = self.local_feature_fused(feature_local)
        return feature_local


class OverlapAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(OverlapAttentionBlock, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.GroupNorm(channels // 32, channels)
        self.act = nn.ReLU()

    def forward(self, x, ol_score):
        '''
        :param x: (B, C, N)
        :param ol: (B, N)
        :return: (B, C, N)
        '''
        B, C, N = x.size()
        x_q = self.q_conv(x).permute(0, 2, 1).contiguous() # B, N, C
        x_k = self.k_conv(x) # B, C, N
        x_v = self.v_conv(x)
        attention = torch.bmm(x_q, x_k) # B, N, N
        if ol_score is not None:
            ol_score = torch.unsqueeze(ol_score, dim=-1).repeat(1, 1, N) # (B, N, N)
            attention = ol_score * attention
        attention = torch.softmax(attention, dim=-1)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention) # B, C, N
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class OverlapAttention(nn.Module):
    def __init__(self, dim):
        super(OverlapAttention, self).__init__()
        self.overlap_attention1 = OverlapAttentionBlock(dim)
        self.overlap_attention2 = OverlapAttentionBlock(dim)
        self.overlap_attention3 = OverlapAttentionBlock(dim)
        self.overlap_attention4 = OverlapAttentionBlock(dim)
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(dim*4, dim*4, kernel_size=1, bias=False),
            nn.GroupNorm(16, dim*4),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, ol):
        x1 = self.overlap_attention1(x, ol)
        x2 = self.overlap_attention2(x1, ol)
        x3 = self.overlap_attention3(x2, ol)
        x4 = self.overlap_attention4(x3, ol)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_fuse(x)
        return x


class TFMRModule(nn.Module):
    def __init__(self):
        super(TFMRModule, self).__init__()
        self.N1s = [512, 512]
        self.M1s = [512, 512]
        self.top_probs = [0.6,0.6]
        self.similarity_topks = [3,1]
        self.use_ppf = False
        in_dim = 6
        self.local_features = LocalFeatue(radius=0.3,
                                          K=64,
                                          in_dim=in_dim,
                                          out_dims=[128, 256, 192])
        self.overlap_attention = OverlapAttention(192)

        # to reuse features for tgt
        self.tgt_info = {'f_y_atten': None,
                         'tgt': None}

    def forward(self, src, tgt, x_ol_score, train, iter,
                normal_src=None, normal_tgt=None):
        '''

        :param src: (B, N, 3)
        :param tgt: (B, M, 3)
        :param x_ol_score: (B, N)
        :param y_ol_score: (B, M)
        :param train: bool, for hyperparameters selection
        :param iter: int
        :param normal_src: (B, N, 3)
        :param normal_tgt: (B, M, 3)
        :return: src: (B, N2, 3); tgt_corr: (B, N2, 3); icp_weights: (B, N2);
        similarity_max_inds: (B, N2), for overlap evaluation.
        '''
        '''

        :param src: 
        :param tgt: 
        :param x_ol_score: 
        return: correspondences points: (N2, 3), (N2, 3)
        '''
        B, _, _ = src.size()
        if train:
            N1, M1, top_prob = self.N1s[0], self.M1s[0], self.top_probs[0]
            similarity_topk = self.similarity_topks[0]
        else:
            N1, M1, top_prob = self.N1s[1], self.M1s[1], self.top_probs[1]
            similarity_topk = self.similarity_topks[1]

        # point feature extraction for tgt
        if self.use_ppf and normal_src is not None:
            f_x_p = self.local_features(feature=normal_src,
                                        xyz=src,
                                        use_ppf=True)
        else:
            f_x_p = self.local_features(feature=None,
                                        xyz=src)

        f_x_attn = self.overlap_attention(f_x_p, ol=None). \
            permute(0, 2, 1).contiguous()
        f_x_atten = f_x_attn / (torch.norm(f_x_attn, dim=-1,
                                        keepdim=True) + 1e-8)  # (B, N, C)
        # x_ol_score, x_ol_inds = torch.sort(x_ol_score, dim=-1, descending=True)
        # x_ol_inds = x_ol_inds[:, :N1]
        # f_x_atten = gather_points(f_x_atten, x_ol_inds)  # (B, N1, C)
        # src = gather_points(src, x_ol_inds)

        if iter == 0:
            # point feature extraction for tgt
            if self.use_ppf and normal_tgt is not None:
                f_y_p = self.local_features(feature=normal_tgt,
                                            xyz=tgt,
                                            use_ppf=True)
            else:
                f_y_p = self.local_features(feature=None,
                                            xyz=tgt)

            f_y_atten = self.overlap_attention(f_y_p, ol=None). \
                permute(0, 2, 1).contiguous()
            f_y_atten = f_y_atten / (torch.norm(f_y_atten, dim=-1,
                                                keepdim=True) + 1e-8)  # (B, M, C)
            # y_ol_score, y_ol_inds = torch.sort(y_ol_score, dim=-1, descending=True)
            # y_ol_inds = y_ol_inds[:, :M1]
            # f_y_atten = gather_points(f_y_atten, y_ol_inds)  # (B, M1, C)
            # tgt = gather_points(tgt, y_ol_inds)

            self.tgt_info = {'f_y_atten': f_y_atten,
                             'tgt': tgt}
        else:
            f_y_atten, tgt = self.tgt_info['f_y_atten'], self.tgt_info['tgt']

        similarity = torch.bmm(f_x_atten, f_y_atten.permute(0, 2, 1).contiguous()) # (B, N1, M1)

        # feature matching removal
        N2 = int(top_prob * N1)  # train
        similarity_max = torch.max(similarity, dim=-1)[0]  # (B, N1)
        similarity_max_inds = \
            torch.sort(similarity_max, dim=-1, descending=True)[1][:, :N2]
        src = gather_points(src, similarity_max_inds)

        # generate correspondences
        similarity = gather_points(similarity, similarity_max_inds) # (B, N2, M1)
        x_ol_score = torch.squeeze(
            gather_points(torch.unsqueeze(x_ol_score, dim=-1),
                          similarity_max_inds), dim=-1)
        # find topk points in feature space
        device = similarity.device
        similarity_topk_inds = \
            torch.topk(similarity, k=similarity_topk, dim=-1)[1]  # (B, N2, topk)
        mask = torch.zeros_like(similarity).to(device).detach()
        inds1 = torch.arange(B, dtype=torch.long).to(device). \
            reshape((B, 1, 1)).repeat((1, N2, similarity_topk))
        inds2 = torch.arange(N2, dtype=torch.long).to(device). \
            reshape((1, N2, 1)).repeat((B, 1, similarity_topk))
        mask[inds1, inds2, similarity_topk_inds] = 1
        similarity = similarity * mask

        weights = similarity / \
                  (torch.sum(similarity, dim=-1, keepdim=True) + 1e-8)
        tgt_corr = torch.bmm(weights, tgt)
        icp_weights = x_ol_score[:, :N2]
        return src, tgt_corr, icp_weights, similarity_max_inds

def compute_rigid_transformation(src, src_corr, weight):
    """
        Compute rigid transforms between two point sets
        Args:
            src: Source point clouds. Size (B, 3, N)
            src_corr: Pseudo target point clouds. Size (B, 3, N)
            weights: Inlier confidence. (B, 1, N)

        Returns:
            R: Rotation. Size (B, 3, 3)
            t: translation. Size (B, 3, 1)
    """
    src2 = (src * weight).sum(dim = 2, keepdim = True) / weight.sum(dim = 2, keepdim = True)
    src_corr2 = (src_corr * weight).sum(dim = 2, keepdim = True)/weight.sum(dim = 2,keepdim = True)
    src_centered = src - src2
    src_corr_centered = src_corr - src_corr2
    H = torch.matmul(src_centered * weight, src_corr_centered.transpose(2, 1).contiguous())

    R = []

    for i in range(src.size(0)):
        u, s, v = torch.svd(H[i])
        r = torch.matmul(v, u.transpose(1, 0)).contiguous()
        r_det = torch.det(r).item()
        diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                            [0, 1.0, 0],
                                            [0, 0, r_det]]).astype('float32')).to(v.device)
        r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
        R.append(r)

    R = torch.stack(R, dim = 0).cuda()

    t = torch.matmul(-R, src2.mean(dim = 2, keepdim=True)) + src_corr2.mean(dim = 2, keepdim = True)
    return R, t

def gather_points(points, inds):
    '''

    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.num_keypoints = 256
        self.weight_function = Discriminator()
        self.fuse = Pointer()
        self.nn_margin = 0.7
        
    def forward(self, *input):
        """
            Args:
                src: Source point clouds. Size (B, 3, N)
                tgt: target point clouds. Size (B, 3, M)
                src_embedding: Features of source point clouds. Size (B, C, N)
                tgt_embedding: Features of target point clouds. Size (B, C, M)
                src_idx: Nearest neighbor indices. Size [B * N * k]
                k: Number of nearest neighbors.
                src_knn: Coordinates of nearest neighbors. Size [B, N, K, 3]
                i: i-th iteration.
                tgt_knn: Coordinates of nearest neighbors. Size [B, M, K, 3]
                src_idx1: Nearest neighbor indices. Size [B * N * k]
                idx2:  Nearest neighbor indices. Size [B, M, k]
                k1: Number of nearest neighbors.
            Returns:
                R/t: rigid transformation.
                src_keypoints, tgt_keypoints: Selected keypoints of source and target point clouds. Size (B, 3, num_keypoint)
                src_keypoints_knn, tgt_keypoints_knn: KNN of keypoints. Size [b, 3, num_kepoints, k]
                loss_scl: Spatial Consistency loss.
        """
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        src_idx = input[4]
        k = input[5]
        src_knn = input[6] # [b, n, k, 3]
        i = input[7]
        tgt_knn = input[8] # [b, n, k, 3]
        src_idx1 = input[9] # [b * n * k1]
        idx2 = input[10] #[b, m, k1]
        k1 = input[11]

        batch_size, num_dims_src, num_points = src.size()
        batch_size, _, num_points_tgt = tgt.size()
        batch_size, _, num_points = src_embedding.size()

        ########################## Matching Map Refinement Module ##########################
        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding) #[b, n, m]
        # point-wise matching map
        scores = torch.softmax(-distance_map, dim=2) #[b, n, m]  Eq. (1)
        # neighborhood-wise matching map
        src_knn_scores = scores.view(batch_size * num_points, -1)[src_idx1, :]
        src_knn_scores = src_knn_scores.view(batch_size, num_points, k1, num_points_tgt) # [b, n, k, m]
        src_knn_scores = pointnet2_utils.gather_operation(src_knn_scores.view(batch_size * num_points, k1, num_points_tgt),\
            idx2.view(batch_size, 1, num_points_tgt * k1).repeat(1, num_points, 1).view(batch_size * num_points, num_points_tgt * k1).int()).view(batch_size,\
                num_points, k1, num_points_tgt, k1)[:, :, 1:, :, 1:].sum(-1).sum(2) / (k1-1) # Eq. (2)

        src_knn_scores = self.nn_margin - src_knn_scores
        refined_distance_map = torch.exp(src_knn_scores) * distance_map
        refined_matching_map = torch.softmax(-refined_distance_map, dim=2) # [b, n, m] Eq. (3)

        # pseudo correspondences of source point clouds (pseudo target point clouds)
        src_corr = torch.matmul(tgt, refined_matching_map.transpose(2, 1).contiguous())# [b,3,n] Eq. (4)

        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, num_points, k, num_dims_src)#[b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods 
        knn_distance = src_corr.transpose(2,1).contiguous().unsqueeze(2) - src_knn_corr #[b, n, k, 3]
        src_knn_distance = src.transpose(2,1).contiguous().unsqueeze(2) - src_knn #[b, n, k, 3]
        
        # inlier confidence
        weight = self.weight_function(knn_distance, src_knn_distance)#[b, 1, n] # Eq. (7)

        # compute rigid transformation 
        R, t = compute_rigid_transformation(src, src_corr, weight) # weighted SVD

        ########################### Preparation for the Loss Function #########################
        # choose k keypoints with highest weights
        src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, self.num_keypoints)

        # spatial consistency loss 
        idx_tgt_corr = torch.argmax(refined_matching_map, dim=-1).int() # [b, n]
        identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1) # [b, m, m]
        one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]
        src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1) # [b, m, num_keypoints]
        keypoints_one_hot = torch.gather(one_hot_number, dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        #[b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
        predicted_keypoints_scores = torch.gather(refined_matching_map.transpose(2, 1), dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * self.num_keypoints, num_points_tgt)
        loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

        # neighorhood information
        src_keypoints_idx2 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, k) #[b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]

        src_transformed = transform_point_cloud(src, R, t.view(batch_size, 3))
        src_transformed_knn_corr = src_transformed.transpose(2,1).contiguous().view(batch_size * num_points, -1)[src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, num_points, k, num_dims_src) #[b, n, k, 3]

        knn_distance2 = src_transformed.transpose(2,1).contiguous().unsqueeze(2) - src_transformed_knn_corr #[b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0,3,1,2), dim = 2, index = src_keypoints_idx2) #[b, 3, num_kepoints, k]
        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn, loss_scl

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.criterion2 = ChamferLoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.GAL = GlobalAlignLoss()
        self.margin = 0.01

    def forward(self, *input):
        """
            Compute global alignment loss and neighorhood consensus loss
            Args:
                src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoint)
                tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoint)
                rotation_ab: Size (B, 3, 3)
                translation_ab: Size (B, 3)
                src_keypoints_knn: [b, 3, num_kepoints, k]
                tgt_keypoints_knn: [b, 3, num_kepoints, k]
                k: Number of nearest neighbors.
                src_transformed: Transformed source point clouds. Size (B, 3, N)
                tgt: Target point clouds. Size (B, 3, M)
            Returns:
                neighborhood_consensus_loss
                global_alignment_loss
        """
        src_keypoints = input[0]
        tgt_keypoints = input[1]
        rotation_ab = input[2]
        translation_ab = input[3]
        src_keypoints_knn = input[4]
        tgt_keypoints_knn = input[5]
        k = input[6]
        src_transformed = input[7]
        tgt = input[8]

        batch_size = src_keypoints.size()[0]

        global_alignment_loss = self.GAL(src_transformed.permute(0, 2, 1), tgt.permute(0, 2, 1), self.margin) 
        
        transformed_srckps_forward = transform_point_cloud(src_keypoints, rotation_ab, translation_ab)
        keypoints_loss = self.criterion(transformed_srckps_forward, tgt_keypoints)
        knn_consensus_loss = self.criterion(src_keypoints_knn, tgt_keypoints_knn)
        neighborhood_consensus_loss = knn_consensus_loss/k + keypoints_loss

        return neighborhood_consensus_loss, global_alignment_loss

class SVD(nn.Module):
    def __init__(self, emb_dims, input_shape="bnc"):
        super(SVD, self).__init__()
        self.emb_dims = emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.input_shape = input_shape

    def forward(self, source, template):
        batch_size = source.size(0)
        if self.input_shape == "bnc":
            source = source.permute(0, 2, 1)  # [B,3,N]
            template = template.permute(0, 2, 1)

        # 经过一个pose变换后的源点云Pt'
        tentative_transform = source

        source_centered = source - source.mean(dim=2, keepdim=True)
        tentative_transform_centered = tentative_transform - tentative_transform.mean(dim=2, keepdim=True)

        H = torch.matmul(source_centered, tentative_transform_centered.permute(0, 2, 1).contiguous())
        U, S, V = [], [], []
        R = []

        for i in range(source.size(0)):
            u, s, v = torch.svd(H[i].data)
            # u, s, v = torch.linalg.svd(H[i])
            u = u.to(source.device)
            s = s.to(source.device)
            v = v.to(source.device)
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            reflect = self.reflect
            reflect = reflect.to(source.device)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                r = r * reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)
        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, source.mean(dim=2, keepdim=True)) + tentative_transform.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)
    
def pairwise_distance_batch(x,y):
    """ 
        pairwise_distance
        Args:
            x: Input features of source point clouds. Size [B, c, N]
            y: Input features of source point clouds. Size [B, c, M]
        Returns:
            pair_distances: Euclidean distance. Size [B, N, M]
    """
    xx = torch.sum(torch.mul(x,x), 1, keepdim = True)#[b,1,n]
    yy = torch.sum(torch.mul(y,y),1, keepdim = True) #[b,1,n]
    inner = -2*torch.matmul(x.transpose(2,1),y) #[b,n,n]
    pair_distance = xx.transpose(2,1) + inner + yy #[b,n,n]
    device = torch.device('cuda')
    zeros_matrix = torch.zeros_like(pair_distance,device = device)
    pair_distance_square = torch.where(pair_distance > 0.0,pair_distance,zeros_matrix)
    error_mask = torch.le(pair_distance_square,0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float()*1e-16)
    pair_distances = torch.mul(pair_distances,(1.0-error_mask.float()))
    return pair_distances

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k):
    """ 
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            relative_coordinates: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
            knn_points: Coordinates of nearest neighbors. Size[B, N, K, 3].
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    knn_points = x.view(batch_size * num_points, -1)[idx, :]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims)#[b, n, k, 3],knn
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)#[b, n, k, 3],central points

    relative_coordinates = (knn_points - x).permute(0, 3, 1, 2)

    return idx, relative_coordinates, knn_points, idx2

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

class Pointer(nn.Module):
    def __init__(self):
        super(Pointer, self).__init__()
        self.conv1 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(256, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return x

def get_knn_index(x, k):
    """ 
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    return idx, idx2



def get_keypoints(src, src_corr, weight, num_keypoints):
    """
        Compute rigid transforms between two point sets
        Args:
            src: Source point clouds. Size (B, 3, N)
            src_corr: Pseudo target point clouds. Size (B, 3, N)
            weights: Inlier confidence. (B, 1, N)
            num_keypoints: Number of selected keypoints.

        Returns:
            src_topk_idx: Keypoint indices. Size (B, 1, num_keypoints)
            src_keypoints: Keypoints of source point clouds. Size (B, 3, num_keypoints)
            tgt_keypoints: Keypoints of target point clouds. Size (B, 3, num_keypoints)
    """
    src_topk_idx = torch.topk(weight, k = num_keypoints, dim = 2, sorted=False)[1]
    src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
    src_keypoints = torch.gather(src, dim = 2, index = src_keypoints_idx)
    tgt_keypoints = torch.gather(src_corr, dim = 2, index = src_keypoints_idx)
    return src_topk_idx, src_keypoints, tgt_keypoints
    

class DGCNN(nn.Module):
    def __init__(self, emb_dims=256):
        super(DGCNN, self).__init__()
        self.emb_dims = emb_dims
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(256, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x):
        """ 
            Simplified DGCNN.
            Args:
                x: Relative coordinates between nearest neighbors and the center point. Size [B, 3, N, K]
            Returns:
                x: Features. Size [B, self.emb_dims, N]
        """
        batch_size, num_dims, num_points, _ = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

class feature_extractor(nn.Module):
    def __init__(self, emb_dims=256):
        super(feature_extractor, self).__init__()
        self.model = DGCNN(emb_dims)

    def forward(self, x, k):
        """ 
            feature extraction.
            Args:
                x: Input point clouds. Size [B, 3, N]
                k: Number of nearest neighbors.
            Returns:
                features: Size [B, C, N]
                idx: Nearest neighbor indices. Size [B * N * k]
                knn_points: Coordinates of nearest neighbors Size [B, N, K, 3].
                idx2: Nearest neighbor indices. Size [B, N, k]
        """
        batch_size, num_dims, num_points = x.size()
        idx, relative_coordinates, knn_points, idx2 = get_graph_feature(x,k)
        features = self.model(relative_coordinates)
        return features, idx, knn_points, idx2

class Discriminator(nn.Module):
    def __init__(self, dim=256):
        super(Discriminator, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=(3,1), bias=True, padding=(1,0)),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim * 2, kernel_size=(3,1), bias=True, padding=(1,0)),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.model2 = nn.Sequential(
            nn.Conv2d(dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.model3 = nn.Sequential(
            nn.Conv2d(dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))

        self.model4 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=1, bias=True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True),
            #nn.Tanh(),
        )

        self.tah = nn.Tanh()

    def forward(self, x, y):
        """ 
            Inlier Evaluation.
            Args:
                x: Source neighborhoods. Size [B, N, K, 3]
                y: Pesudo target neighborhoods. Size [B, N, K, 3]
            Returns:
                x: Inlier confidence. Size [B, 1, N]
        """
        b, n, k, _ = x.size()

        x_1x3 = self.model1(x.permute(0,3,2,1)).permute(0,1,3,2)

        y_1x3 = self.model1(y.permute(0,3,2,1)).permute(0,1,3,2) # [b, n, k, 3]-[b, c, k, n]-->[b, c, n, k]
        
        x2 = x_1x3 - y_1x3 # Eq. (5)
        
        x = self.model2(x2) # [b, c, n, k]
        weight = self.model3(x2) # [b, c, n, k] 
        weight = torch.softmax(weight, dim=-1) # Eq. (6)
        x = (x * weight).sum(-1) # [b, c, n]
        x = 1 - self.tah(torch.abs(self.model4(x)))
        return x
def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha

def batch_transform(batch_pc, batch_R, batch_t=None):
    '''
    :param batch_pc: shape=(B, N, 3)
    :param batch_R: shape=(B, 3, 3)
    :param batch_t: shape=(B, 3)
    :return: shape(B, N, 3)
    '''
    transformed_pc = torch.matmul(batch_pc, batch_R.permute(0, 2, 1).contiguous())
    if batch_t is not None:
        transformed_pc = transformed_pc + torch.unsqueeze(batch_t, 1)
    return transformed_pc


def batch_quat2mat(batch_quat):
    '''
    :param batch_quat: shape=(B, 4)
    :return:
    '''
    batch_quat = batch_quat.squeeze()
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], \
                 batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=torch.float, size=(B, 3, 3)).to(device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R