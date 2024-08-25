import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pointnet2_ops import pointnet2_utils
import copy
import math, sys
sys.path.append("..")
from feature_extract import AGCNN,AGCNN_IA
from chamfer_loss import *
from utils import *



class MLPs(nn.Module):
    def __init__(self, in_dim, mlps):
        super(MLPs, self).__init__()
        self.mlps = nn.Sequential()
        l = len(mlps)
        for i, out_dim in enumerate(mlps):
            self.mlps.add_module(f'fc_{i}', nn.Linear(in_dim, out_dim))
            if i != l - 1:
                self.mlps.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.mlps(x)
        return x


def mask_point(mask_idx, points):
    # masks: [b, n] : Tensor, 包含0和1
    # points: [b, 3, n] : Tensor
    # return: [b, 3, n2] : Tensor
    batch_size = points.shape[0]
    points = points.permute(0, 2, 1)
    #mask_idx = mask_idx.reshape(batch_size, -1, 1)
    new_pcs = points #* mask_idx #
    new_points = []
    i = 0
    for new_pc in new_pcs:

        # 删除被屏蔽的0点
        temp = mask_idx[i] == 0
        temp = temp.cpu()
        idx = np.argwhere(temp)
        new_point = np.delete(new_pc.cpu().detach().numpy(), idx, axis=0)

        new_points.append(new_point)
        i+=1

    new_points = np.array(new_points, dtype=np.float)
    new_points = torch.Tensor(new_points).cuda()
    return new_points.permute(0, 2, 1)

def mask_point_padding(mask_idx, points):
    # masks: [b, n] : Tensor, 包含0和1
    # points: [b, 3, n] : Tensor
    # return: [b, 3, n2] : Tensor
    batch_size,num_dims,num_points = points.shape
    points = points.permute(0, 2, 1)
    #mask_idx = mask_idx.reshape(batch_size, -1, 1)
    new_pcs = points #* mask_idx #
    new_points = []
    i = 0
    for new_pc in new_pcs:

        # 删除被屏蔽的0点
        temp = mask_idx[i] == 0
        temp = temp.cpu()
        idx = np.argwhere(temp)
        new_point = np.delete(new_pc.cpu().detach().numpy(), idx, axis=0)
        zero_point = 1e-6*np.ones((idx.shape[1],num_dims))
        new_point = np.concatenate((new_point,zero_point),axis=0)
        new_points.append(new_point)
        i+=1

    new_points = np.array(new_points, dtype=np.float)
    new_points = torch.Tensor(new_points).cuda()
    return new_points.permute(0, 2, 1)


def overlap_point(src_embedding, tgt_embedding):
    batch_size, num_dims, num_points1 = src_embedding.size()
    batch_size, num_dims, num_points2 = tgt_embedding.size()
    if num_points1 >= num_points2:
        src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
        tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
        cos_simi = torch.matmul(tar_norm.transpose(2, 1).contiguous(), src_norm)  # (batch, num_points2, num_points1)
        cos_simi = torch.max(cos_simi, dim=1).values
        _, idx1 = torch.sort(cos_simi, descending=True)
        idx = idx1[:, :num_points2]
        idx_oh = F.one_hot(idx, num_points1)
        idx_oh = torch.sum(idx_oh, dim=1)
        return idx_oh, False

    elif num_points1 < num_points2:
        src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
        tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
        cos_simi = torch.matmul(tar_norm.transpose(2, 1).contiguous(), src_norm)  # (batch, num_points2, num_points1)
        cos_simi = torch.max(cos_simi, dim=2).values
        _, idx1 = torch.sort(cos_simi, descending=True)
        idx = idx1[:, :num_points1]
        idx_oh = F.one_hot(idx, num_points2)
        idx_oh = torch.sum(idx_oh, dim=1)
        return idx_oh, True

class OverlapNet(nn.Module):
    def __init__(self, num_subsampled_points, n_emb_dims=512, k=20):
        super(OverlapNet, self).__init__()
        self.emb_dims = n_emb_dims
        self.k = k
        # self.init_reg = InitReg()
        self.emb_nn = AGCNN(n_emb_dims=self.emb_dims,k=self.k)
        # self.emb_nn = DGCNN(n_emb_dims=self.emb_dims)
        self.emb_ia = AGCNN_IA(n_emb_dims=self.emb_dims,k=self.k)
        # 第二个点云的点数(缺失点云的点数)
        self.num_subsampled_points = num_subsampled_points
        self.mask_nn = nn.Sequential(nn.Conv1d(self.num_subsampled_points, 1024, 1), nn.BatchNorm1d(1024), nn.LeakyReLU(negative_slope=0.01),
                                nn.Conv1d(1024, 512, 1), nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.01),
								nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
                                nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.LeakyReLU(negative_slope=0.01),
								nn.Conv1d(128, 1, 1), nn.Sigmoid())
        self.sigm = nn.Sigmoid()

    def forward(self, *input):
        src = input[0]  # 1024
        tgt = input[1]  # 768
        batch_size = src.shape[0]

        src_embedding, src_ia = self.emb_nn(src)
        tgt_embedding = self.emb_ia(tgt, src_ia)

        batch_size, num_dims, num_points1 = src_embedding.size()
        batch_size, num_dims, num_points2 = tgt_embedding.size()

        src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
        tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
        cos_simi = torch.matmul(tar_norm.transpose(2, 1).contiguous(),
                              src_norm)  # (batch, num_points2, num_points1)
        # glob_residual = torch.abs(src_glob - tar_glob)
        # threshold = self.threshold_nn(glob_residual)
        # threshold = 0.5
        mask = self.mask_nn(cos_simi).reshape(batch_size, -1)
        # mask_idx = torch.where(mask >= threshold, 1, 0)

        _, idx1 = torch.sort(mask,descending=True)
        # rate = float(torch.mean(overlap_rate).cpu().detach().numpy())
        # sample_point = int(rate * 1024)
        idx = idx1[:, :self.num_subsampled_points]
        idx_oh = F.one_hot(idx,num_points1)
        idx_oh = torch.sum(idx_oh,dim=1)

        return mask, idx_oh,src_embedding,tgt_embedding
        # return batch_R, batch_t, mask, idx_oh, over_idx, flag
        
class SelfEncoder(nn.Module):
    def __init__(self,  k_nn=20):
        super().__init__()
        self.k = k_nn
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1, bias=False),self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64*2, 128, kernel_size=1, bias=False),self.bn3,nn.LeakyReLU(negative_slope=0.2))

    def _get_graph_feature(self, x, k=20, idx=None):

        def knn(x, k):
            inner = -2*torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x**2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            s,idx = pairwise_distance.topk(k=k, dim=-1)
            return idx

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = knn(x, k=k)
        device = idx.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
        return feature

    def cos_simi(self, x, nn_feature):
        # x: (batch_size, dims, num_points)
        # nn_feature: (batch_size, num_points, k, f_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, f_dim, num_points)
        x = x.permute(0, 3, 1, 2)  # (batch_size, num_points, 1, f_dim)

        x_norm = F.normalize(x, p=2, dim=3) + 1e-5
        nn_feature_norm = F.normalize(nn_feature, p=2, dim=3) + 1e-5
        # print(x_norm.shape, nn_feature_norm.shape)
        simi = torch.matmul(x_norm, nn_feature_norm.permute(0, 1, 3, 2))  # (batch, num_points, 1, k)

        return simi

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        distance = -xx - inner - xx.transpose(2, 1).contiguous()
        s,idx = distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
        # distance = cos_simi(x, x)

        return idx,s

    def graph_self_attention(self, x, k=20):
        # (batch, f_dim, num_points)
        x = x.view(*x.size()[:3])
        idx,distance = self.knn(x, k=k)  # (batch_size, num_points, k)
        batch_size, num_points, _ = idx.size()
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]

        nn_feature = feature.view(batch_size, num_points, k, num_dims)
        x_ = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        nn_feature = nn_feature - x_
        
        weight = torch.softmax(distance,dim=-1).unsqueeze(-1).repeat(1, 1, 1, num_dims)
        weight_nn_feature = (weight*nn_feature).sum(dim=2)
        # simi = self.cos_simi(x.permute(0, 2, 1), nn_feature)  # (batch, num_points, 1, k)
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze()  # (batch, num_points, f_dim)
        attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        return attent_feature.permute(0, 2, 1)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = self.graph_self_attention(x, self.k)
        x1 = self.conv1(x)

        x = self.graph_self_attention(x1, self.k)
        x2 = self.conv2(x)

        x = self.graph_self_attention(x2, self.k)
        x3 = self.conv3(x)
        return x1, x2, x3

class CrossEncoder(nn.Module):
    def __init__(self, enc_emb_dim=128, enc_glb_dim=1024, k_nn=20):
        super().__init__()
        self.k = k_nn
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.GroupNorm(32,512)
        self.bn6 = nn.GroupNorm(32,512)
        self.bn7 = nn.GroupNorm(32,256)
        self.bn8 = nn.GroupNorm(32,enc_emb_dim)
        self.conv1 = nn.Sequential(nn.Conv1d(6, 64, kernel_size=1, bias=False),self.bn1,nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=False),self.bn2,nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(64*2, 128, kernel_size=1, bias=False),self.bn3,nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128*2, 256, kernel_size=1, bias=False),self.bn4,nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, enc_glb_dim//2, kernel_size=1, bias=False),self.bn5,nn.LeakyReLU(negative_slope=0.2))
        self.mlp = nn.Sequential(
            nn.Conv1d(64+64+128+256, 512, 1),#+enc_glb_dim
            self.bn6,
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            self.bn7,
            nn.ReLU(),
            nn.Conv1d(256, enc_emb_dim, 1),
            self.bn8,
            nn.ReLU()
            )

    def cos_simi(self, x, nn_feature):
        # x: (batch_size, dims, num_points)
        # nn_feature: (batch_size, num_points, k, f_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, f_dim, num_points)
        x = x.permute(0, 3, 1, 2)  # (batch_size, num_points, 1, f_dim)

        x_norm = F.normalize(x, p=2, dim=3) + 1e-5
        nn_feature_norm = F.normalize(nn_feature, p=2, dim=3) + 1e-5
        # print(x_norm.shape, nn_feature_norm.shape)
        simi = torch.matmul(x_norm, nn_feature_norm.permute(0, 1, 3, 2))  # (batch, num_points, 1, k)

        return simi

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        distance = -xx - inner - xx.transpose(2, 1).contiguous()
        s,idx = distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
        # distance = cos_simi(x, x)

        return idx,s

    def cos_simi_ds(self, src_embedding, tgt_embedding):
        # (batch, emb_dims, num_points)
        # src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
        # tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
        src_norm = F.normalize(src_embedding, p=2, dim=1)
        tar_norm = F.normalize(tgt_embedding, p=2, dim=1)
        simi = torch.matmul(src_norm.transpose(2, 1).contiguous(), tar_norm)  # (batch, num_points1, num_points2)
        return simi

    def knn_IA(self,x, y, k):
        # inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), y)
        # xx = torch.sum(x ** 2, dim=1, keepdim=True)
        # distance = -xx - inner - xx.transpose(2, 1).contiguous()
        distance = self.cos_simi_ds(x, y)
        s,idx = distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

        return idx, distance.size()[2],s

    def graph_self_attention(self, x, k):
        # (batch, f_dim, num_points)
        x = x.view(*x.size()[:3])
        idx,distance = self.knn(x, k=k)  # (batch_size, num_points, k)
        batch_size, num_points, _ = idx.size()
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = x.size()
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]

        nn_feature = feature.view(batch_size, num_points, k, num_dims)
        x_ = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
        nn_feature = nn_feature - x_

        weight = torch.softmax(distance,dim=-1).unsqueeze(-1).repeat(1, 1, 1, num_dims)
        weight_nn_feature = (weight*nn_feature).sum(dim=2)
        # simi = self.cos_simi(x.permute(0, 2, 1), nn_feature)  # (batch, num_points, 1, k)
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze()  # (batch, num_points, f_dim)
        attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        return attent_feature.permute(0, 2, 1)


    def graph_cross_attention(self, x, y, k):
        # (batch, f_dim, num_points)
        x = x.view(*x.size()[:3])
        y = y.view(*y.size()[:3])
        idy,num_points2,distance = self.knn_IA(x, y, k=k)  # (batch_size, num_points, k)
        batch_size, num_points1, _ = idy.size()

        idy_base = torch.arange(0, batch_size, device=y.device).view(-1, 1, 1) * num_points2
        idy = idy + idy_base
        idy = idy.view(-1)
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        y = y.transpose(2, 1).contiguous()
        feature = y.view(batch_size * num_points2, -1)[idy, :]

        nn_feature = feature.view(batch_size, num_points1, k, num_dims)
        x_ = x.view(batch_size, num_points1, 1, num_dims).repeat(1, 1, k, 1)
        nn_feature = nn_feature - x_

        weight = torch.softmax(distance,dim=-1).unsqueeze(-1).repeat(1, 1, 1, num_dims)
        weight_nn_feature = (weight*nn_feature).sum(dim=2)
        # simi = self.cos_simi(x.permute(0, 2, 1), nn_feature)  # (batch, num_points, 1, k)
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze()  # (batch, num_points, f_dim)
        attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        return attent_feature.permute(0, 2, 1)

    def forward(self, x1,x2,x3, y3):
        batch_size, _, num_points = x1.size()

        x = self.graph_cross_attention(x3, y3, self.k)
        x4 = self.conv4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01).view(batch_size, -1, num_points)
        # return x

        x = self.conv5(x)
        local_concat = x
        # embedding_feat = x
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        global_vector = x
        repeat_glb_feat = global_vector.unsqueeze(-1).expand(batch_size, global_vector.shape[1], num_points)
        x = torch.cat((local_concat, repeat_glb_feat), 1)
        embedding_feat = self.mlp(local_concat)
        return embedding_feat, global_vector.unsqueeze(-1)

class Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input*8
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class norm(nn.Module):
    def __init__(self, axis=2):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        mean = torch.mean(x, self.axis,keepdim=True)
        std = torch.std(x, self.axis,keepdim=True)
        x = (x-mean)/(std+1e-6)
        return x

class Modified_softmax(nn.Module):
    def __init__(self, axis=1):
        super(Modified_softmax, self).__init__()
        self.axis = axis
        self.norm = norm(axis = axis)
    def forward(self, x):
        x = self.norm(x)
        x = Gradient.apply(x)
        x = F.softmax(x, dim=self.axis)
        return x

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)  # Flatten the inputs
        targets = targets.view(-1)  # Flatten the targets

        # Compute the log-probabilities
        log_probs = F.log_softmax(inputs, dim=0)

        # Gather the log-probabilities for the targets
        log_probs = log_probs.gather(0, targets)

        # Compute the focal loss
        loss = -self.alpha * (1 - torch.exp(log_probs)) ** self.gamma * log_probs

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class PEAL(nn.Module):
    def __init__(self, n_emb_dims=512,input_pts=768,enc_emb_dim=128, enc_glb_dim=1024, filetype='modelnet40'):
        super(PEAL, self).__init__()
        self.overlap = OverlapNet(input_pts,k=20)
        self.input_pts=input_pts
        self.emb_dims = n_emb_dims
        self.enc_emb_dim = enc_emb_dim
        self.enc_glb_dim = enc_glb_dim
        self.dec_in_dim = enc_glb_dim + enc_emb_dim #+3
        self.filetype = filetype
        self.selfencoder = SelfEncoder()
        self.crossencoder = CrossEncoder(self.enc_emb_dim, self.enc_glb_dim)
        self.DeSmooth = nn.Sequential(
            nn.Conv1d(in_channels=int(self.input_pts), out_channels=self.input_pts+128, kernel_size=1, stride=1,  bias=False),
            nn.ReLU(),
            norm(axis=1),
            nn.Conv1d(in_channels=self.input_pts+128, out_channels=int(self.input_pts), kernel_size=1, stride=1,bias=False)
            # Modified_softmax(axis=2)
            )

        self.confi = nn.Sequential(nn.Conv1d(128, 64,kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(64, 32,kernel_size=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(32, 16,kernel_size=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv1d(16, 1,kernel_size=1),
            nn.Sigmoid())
        self.sigm = nn.Sigmoid()
        self.focal_loss = FocalLoss()
        # self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_bce = nn.BCELoss()

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        distance = -xx - inner - xx.transpose(2, 1).contiguous()
        s,idx = distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
        # distance = cos_simi(x, x)

        return idx,s

    def _KFNN(self, x, y, k=10):
        def batched_pairwise_dist(a, b):
            x, y = a.float(), b.float()
            bs, num_points_x, points_dim = x.size()
            bs, num_points_y, points_dim = y.size()
            

            xx = torch.pow(x, 2).sum(2)
            yy = torch.pow(y, 2).sum(2)
            zz = torch.bmm(x, y.transpose(2, 1))
            rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x)
            ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)
            P = rx.transpose(2, 1) + ry - 2 * zz
            return P

        pairwise_distance = batched_pairwise_dist(x.permute(0,2,1), y.permute(0,2,1))
        similarity=-pairwise_distance
        idx = similarity.topk(k=k, dim=-1)[1]
        return pairwise_distance, idx

    def forward(self, source, target,eval=False):
        src = source  # 1024
        tgt = target  # 768
        batch_size, _, num_s = src.shape
        mask, mask_idx,HIER_feat1,HIER_feat2 = self.overlap(src, tgt)
  
        src = mask_point(mask_idx, src)
        olsocre_x = mask_point(mask_idx, mask.unsqueeze(1))
        x1,x2,x3 = self.selfencoder(x=src)
        y1,y2,y3 = self.selfencoder(x=tgt)
        HIER_feat1, pooling_feat1 = self.crossencoder(x1=x1,x2=x2,x3=x3,y3=y3)
        
        HIER_feat2, pooling_feat2 = self.crossencoder(x1=y1,x2=y2,x3=y3,y3=x3)
        # print(HIER_feat1.size()) 
        
        pairwise_distance, _ = self._KFNN(HIER_feat1, HIER_feat2)
        similarity = 1/(pairwise_distance + 1e-6)
        batch_size, num_dims, num_points1 = HIER_feat1.size()
        batch_size, num_dims, num_points2 = HIER_feat2.size()

        p = self.DeSmooth(similarity.transpose(1,2).contiguous()).transpose(1,2).contiguous() 
        weights = F.softmax(sinkhorn(p),dim=2) 
   
        tgt_corr = torch.bmm(weights, tgt.transpose(2,1))
        
        y1_p,y2_p,y3_p = self.selfencoder(x=tgt_corr.transpose(2,1))
        y1_p,y2_p,y3_p = y1_p.detach(),y2_p.detach(),y3_p.detach()
        HIER_feat2_p, _ = self.crossencoder(x1=y1_p,x2=y2_p,x3=y3_p,y3=x3)
        HIER_feat1_p = HIER_feat1.detach()
        HIER_feat2_p= HIER_feat2_p.detach()
        src_norm_p =  HIER_feat1_p / ( HIER_feat1_p.norm(dim=1).reshape(batch_size, 1, num_points1))
        tar_norm_p =  HIER_feat2_p / ( HIER_feat2_p.norm(dim=1).reshape(batch_size, 1, num_points2))
        cos_simi_p = (src_norm_p*tar_norm_p).sum(axis=1)
        cos_simi_net = self.confi(HIER_feat1_p-HIER_feat2_p).squeeze(1)
        

        
        cos_simi_u = cos_simi_p.unsqueeze(1).contiguous() 
        
        if self.filetype == 'modelnet40':
            r=0.01 #modelnet
            r_2 = 0.01**2
            k=10
        elif self.filetype == 'Kitti_odo':
            r=0.6 #KITTI
            r_2 = 0.6**2
            k=25
        elif self.filetype in ['3DMatch', '3DMatch_all']:
            r=0.03 #3dmatch 
            r_2 = 0.02
            k=5
        idx, distance = self.knn(src,k=k)
        batch_size, num_points, _ = idx.size()
        device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        _, num_dims, _ = cos_simi_u.size()
        cos_simi_u = cos_simi_u.transpose(2, 1).contiguous()
        feature = cos_simi_u.view(batch_size * num_points, -1)[idx, :]
        nn_feature = feature.view(batch_size, num_points, k, num_dims)
        weight = torch.softmax(distance,dim=-1).unsqueeze(-1).repeat(1, 1, 1, num_dims)
        cos_simi_weight = (weight*nn_feature).sum(dim=2).squeeze(-1) 
        

        if eval: 
            _, idx1 = torch.sort(cos_simi_net,descending=True)
            idx = idx1[:, :int(num_points2*0.9)]
            idx_oh = F.one_hot(idx,num_points1)
            idx_oh = torch.sum(idx_oh,dim=1)
            tgt_select = idx_oh
        else:
            tgt_select = torch.where(cos_simi_weight>0.9, 1, 0)

        src_s = mask_point_padding(tgt_select, src)
        tgt_corr_s = mask_point_padding(tgt_select, tgt_corr.transpose(2,1).contiguous()) 
        icp_weights = mask_point_padding(tgt_select, olsocre_x)
        

        
        est_R, est_t = compute_rigid_transformation(src_s, tgt_corr_s, icp_weights)
        est_t = est_t.squeeze(2)
        out_a = batch_transform(tgt_corr.transpose(2,1).permute(0, 2, 1), est_R.transpose(2,1).contiguous(), -est_t).permute(0, 2, 1)
        
        dist = ((out_a-src) ** 2).sum(axis=1) 
        if self.input_pts/num_s ==0.4:
            lable = torch.where(dist <r**2, 1.0, 0.0).detach()
            loss_confi = self.loss_bce(cos_simi_net,lable)
        else:
            lable = torch.where(dist <r**2, 1, 0).detach()
            loss_confi = self.focal_loss(cos_simi_net,lable)
        out_b = batch_transform(source.permute(0, 2, 1), est_R.contiguous(), est_t).permute(0, 2, 1) 
        pair_distance,idx_ol = self._KFNN(out_b,target,k=1)
        batch_size, num_points1, _ = idx_ol.size()
        idy_base = torch.arange(0, batch_size, device=target.device).view(-1, 1, 1) * num_points2
        idy = idx_ol + idy_base
        idy = idy.view(-1)
        _, num_dims, _ = out_b.size()
        y = target.transpose(2, 1).contiguous()
        point_ol = y.view(batch_size * num_points2, -1)[idy, :]
        point_ol = point_ol.view(batch_size, num_points1, 1, num_dims).squeeze(2).transpose(2,1).contiguous() 
        dist_ol = ((out_b-point_ol) ** 2).sum(axis=1) 

        lable_ol = torch.where(dist_ol <r_2, 1.0, 0.0).detach() 
        mask_loss = self.loss_bce(mask,lable_ol)
        
        I_N1 = torch.eye(n=p.shape[1], device=device)
        bsize = target.shape[0]
        I_N1 = I_N1.unsqueeze(0).repeat(bsize, 1, 1)
        rank_term = torch.mean(batch_frobenius_norm(torch.bmm(weights, weights.transpose(2, 1).contiguous()), I_N1.float()))
        
        return weights,mask_idx,src,tgt,est_R,est_t,mask_loss,rank_term,loss_confi,lable_ol
  