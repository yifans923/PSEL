import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda, FloatTensor, LongTensor
from typing import Tuple, Callable, Optional, Union
import copy, math
'''
特征提取文件，包含：DGCNN, Transformer, PointNet, PointCNN, PointConv
'''
# torch.cuda.set_device(0)

# ----- DGCNN -----

def cos_simi(src_embedding, tgt_embedding):
    # (batch, emb_dims, num_points)
    # src_norm = src_embedding / (src_embedding.norm(dim=1).reshape(batch_size, 1, num_points1))
    # tar_norm = tgt_embedding / (tgt_embedding.norm(dim=1).reshape(batch_size, 1, num_points2))
    src_norm = F.normalize(src_embedding, p=2, dim=1)
    tar_norm = F.normalize(tgt_embedding, p=2, dim=1)
    simi = torch.matmul(src_norm.transpose(2, 1).contiguous(), tar_norm)  # (batch, num_points1, num_points2)
    return simi

def knn_IA(x, y, k):
    #inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), y)
    #xx = torch.sum(x ** 2, dim=1, keepdim=True)
    #distance = -xx - inner - xx.transpose(2, 1).contiguous()
    distance = cos_simi(x, y)
    s,idx = distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx,distance.size()[2],s



def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()
    s,idx = distance.topk(k=k, dim=-1)  # (batch_size, num_points, k) 
    # distance = cos_simi(x, x)

    return idx,s

def get_graph_feature(x, k=20):
    # x = x.squeeze()
    x = x.view(*x.size()[:3])
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature -= x

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    # [batchsize, 输入特征dim*2, num_points, k]
    return feature

# 特征提取，平均池化
class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        # self.conv = SKConv(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class DGCNN_IA(nn.Module):
    def __init__(self, n_emb_dims=512, k=20):
        super(DGCNN_IA, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

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

    def get_graph_feature_IA(self, x, y, k=20):
        # x = x.squeeze()
        x = x.view(*x.size()[:3])
        y = y.view(*y.size()[:3])
        idy,num_points2 = knn_IA(x, y, k=k)  # (batch_size, num_points, k)
        batch_size, num_points1, _ = idy.size()

        idy_base = torch.arange(0, batch_size, device=y.device).view(-1, 1, 1) * num_points2
        idy = idy + idy_base
        idy = idy.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        y = y.transpose(2, 1).contiguous()
        feature = y.view(batch_size * num_points2, -1)[idy, :]
        feature = feature.view(batch_size, num_points1, k, num_dims)
        x_ = x.view(batch_size, num_points1, 1, num_dims).repeat(1, 1, k, 1)
        feature -= x_

        #simi = self.cos_simi(x.permute(0, 2, 1), feature)  # (batch, num_points, 1, k)
        #weight_nn_feature = (torch.matmul(simi, feature)).squeeze()  # (batch, num_points, f_dim)
        #attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)

        feature = torch.cat((feature, x_), dim=3).permute(0, 3, 1, 2)
        # [batchsize, 输入特征dim*2, num_points, k]
        return feature
        #return attent_feature.permute(0, 2, 1)

    def forward(self, x, y):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x, self.k)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x1, self.k)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2,  self.k)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = self.get_graph_feature_IA(x3, y, self.k)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        # x = self.get_graph_feature_IA(x, y, self.k)
        # x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        # x5 = x.max(dim=-1, keepdim=True)[0].view(batch_size, -1, num_points)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(batch_size, -1, num_points)
        return x

class DGCNN(nn.Module):
    def __init__(self, n_emb_dims=512, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x, self.k)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x1, self.k)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2, self.k)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x3, self.k)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = torch.cat((x1, x2, x3), dim=1)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(batch_size, -1, num_points)


        return x


class AGCNN(nn.Module):
    def __init__(self, n_emb_dims=512, k=20):
        super(AGCNN, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(n_emb_dims)
        # self.mlp1 = nn.Sequential(nn.Conv2d(3, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 3, 1))
        # self.mlp2 = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 64, 1))
        # self.mlp3 = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 64, 1))
        # self.mlp4 = nn.Sequential(nn.Conv2d(128, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 128, 1))


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

    def graph_self_attention(self, x, k):
        # (batch, f_dim, num_points)
        x = x.view(*x.size()[:3])
        idx,distance = knn(x, k=k)  # (batch_size, num_points, k)
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
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze(2)  # (batch, num_points, f_dim)
        attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        return attent_feature.permute(0, 2, 1)

        #weight_nn_feature = torch.matmul(simi, nn_feature)  # (batch, num_points, k，f_dims)
        #feature = weight_nn_feature.view(batch_size, num_points, k, num_dims)
        #feature = torch.cat((feature, x_), dim=3).permute(0, 3, 1, 2)
        #return feature

        # feature = nn_feature.permute(0, 3, 1, 2)
        # B, C, N, M = feature.size()
        # if tag == 1:
        #     mlp = self.mlp1
        # elif tag == 2:
        #     mlp = self.mlp2
        # elif tag == 3:
        #     mlp = self.mlp3
        # elif tag == 4:
        #     mlp = self.mlp4
        # feature = nn_feature.permute(0, 3, 1, 2)
        # feature = feature.transpose(1, 2).contiguous().view(B * N, C, M, 1).repeat(1, 1, 1, M)  # (BN, C, M, M)
        # feature = feature - feature.transpose(2, 3).contiguous() + torch.mul(feature, torch.eye(M).view(1, 1, M, M).cuda())  # (BN, C, M, M)
        # weight = mlp(feature)
        # weight = F.softmax(weight, -1)
        # feature = (feature * weight).sum(-1).view(B, N, C, M).transpose(1, 2).contiguous()  # (B, C, N, M)
        #
        # nn_feature = feature.permute(0, 2, 3, 1)
        # simi = self.cos_simi(x.permute(0, 2, 1), nn_feature)  # (batch, num_points, 1, k)
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze()  # (batch, num_points, f_dim)
        # attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        # return attent_feature.permute(0, 2, 1)

        # feature = torch.cat((x, feature), dim=2)
        # return feature.permute(0, 2, 1)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = self.graph_self_attention(x, self.k)
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        # simi = self.cos_simi(x.permute(0, 2, 1), nn_feature)  # (batch, num_points, 1, k)
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze(2)  # (batch, num_points, f_dim)
        # attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        
        x = self.graph_self_attention(x1, self.k)
        x2 = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)

        x = self.graph_self_attention(x2, self.k)
        x3 = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)

        x = self.graph_self_attention(x3, self.k)
        x4 = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01).view(batch_size, -1, num_points)
        return x, x3
        #return x1, x2, x3

class AGCNN_IA(nn.Module):
    def __init__(self, n_emb_dims=512, k=20):
        super(AGCNN_IA, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(n_emb_dims)
        # self.mlp1 = nn.Sequential(nn.Conv2d(3, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 3, 1))
        # self.mlp2 = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 64, 1))
        # self.mlp3 = nn.Sequential(nn.Conv2d(64, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 64, 1))
        # self.mlp4 = nn.Sequential(nn.Conv2d(128, 16, 1), nn.Conv2d(16, 16, 1), nn.Conv2d(16, 128, 1))

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

    def graph_self_attention(self, x, k):
        # (batch, f_dim, num_points)
        x = x.view(*x.size()[:3])
        idx,distance = knn(x, k=k)  # (batch_size, num_points, k)
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

        #weight_nn_feature = torch.matmul(simi, nn_feature)  # (batch, num_points, k，f_dims)
        #feature = weight_nn_feature.view(batch_size, num_points, k, num_dims)
        #feature = torch.cat((feature, x_), dim=3).permute(0, 3, 1, 2)
        #return feature


    def graph_cross_attention(self, x, y, k):
        # (batch, f_dim, num_points)
        x = x.view(*x.size()[:3])
        y = y.view(*y.size()[:3])
        idy,num_points2,distance = knn_IA(x, y, k=k)  # (batch_size, num_points, k)
        batch_size, num_points1, _ = idy.size()

        idy_base = torch.arange(0, batch_size, device=y.device).view(-1, 1, 1) * num_points2
        idy = idy + idy_base
        idy = idy.view(-1)
        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        y = y.transpose(2, 1).contiguous()
        feature = y.view(batch_size * num_points2, -1)[idy, :]
        # feature = feature.view(batch_size, num_points1, k, num_dims)
        # x_ = x.view(batch_size, num_points1, 1, num_dims).repeat(1, 1, k, 1)
        # feature -= x_

        nn_feature = feature.view(batch_size, num_points1, k, num_dims)
        x_ = x.view(batch_size, num_points1, 1, num_dims).repeat(1, 1, k, 1)
        nn_feature = nn_feature - x_
        
        weight = torch.softmax(distance,dim=-1).unsqueeze(-1).repeat(1, 1, 1, num_dims)
        weight_nn_feature = (weight*nn_feature).sum(dim=2)

        # simi = self.cos_simi(x.permute(0, 2, 1), nn_feature)  # (batch, num_points, 1, k)
        # weight_nn_feature = (torch.matmul(simi, nn_feature)).squeeze()  # (batch, num_points, f_dim)
        attent_feature = torch.cat((x, weight_nn_feature), dim=2)  # (batch, num_points, f_dim*2)
        return attent_feature.permute(0, 2, 1)


    # def forward(self, x1,x2,x3, y3):
    #     batch_size, _, num_points = x1.size()
    #     x = self.graph_cross_attention(x3, y3, self.k)
    #     x4 = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
    #     x = torch.cat((x1, x2, x3, x4), dim=1)
    #     x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01).view(batch_size, -1, num_points)
    #     return x

    def forward(self, x, y3):
        batch_size, num_dims, num_points = x.size()
        x = self.graph_self_attention(x, self.k)
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)

        x = self.graph_self_attention(x1, self.k)
        x2 = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)

        x = self.graph_self_attention(x2, self.k)
        x3 = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)


        x = self.graph_cross_attention(x3, y3, self.k)
        x4 = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01).view(batch_size, -1, num_points)
        return x

class STNkd(nn.Module):
    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(PointNet, self).__init__()
        # self.stn = STNkd(k=64)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, n_emb_dims, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.bn4 = nn.BatchNorm1d(128)
        # self.bn5 = nn.BatchNorm1d(n_emb_dims)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        # trans_feat = self.stn(x2)
        # x2 = x2.transpose(2, 1)
        # x2 = torch.bmm(x2, trans_feat)
        # x2 = x2.transpose(2, 1)
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x = F.relu(self.conv5(x4))
        return x

# ----- PointCNN -----
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]

def knn_indices_func_gpu(rep_pts : cuda.FloatTensor,  # (N, pts, dim)
                         pts : cuda.FloatTensor,      # (N, x, dim)
                         k : int, d : int
                        ) -> cuda.LongTensor:         # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []

    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref)**2, 2).squeeze()
        _, inds = torch.topk(dist2, k*d + 1, dim = 1, largest = False)
        region_idx.append(inds[:,1::d])

    region_idx = torch.stack(region_idx, dim = 0)
    return region_idx


def EndChannels(f, make_contiguous = False):
    """ 改变通道顺序 -> 做卷积 -> 再改回来 """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

class Dense(nn.Module):
    """
    单层mlp，带有激活和dropout
    """

    def __init__(self, in_features : int, out_features : int,
                 drop_rate : int = 0, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        # self.bn = LayerNorm(out_channels) if with_bn else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        # if self.bn:
        #     x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x

class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]], with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.9) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""
    # 做两次卷积

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]],
                 depth_multiplier : int = 1, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.9) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int,
                 P: int, C_mid: int, depth_multiplier: int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()

        if __debug__:
            # Only needed for assertions.
            self.C_in = C_in
            self.C_mid = C_mid
            self.dims = dims
            self.K = K

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers，单层mlp
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)

        # Layers to generate X
        self.x_trans = nn.Sequential(
            EndChannels(Conv(
                in_channels=dims,
                out_channels=K * K,
                kernel_size=(1, K),
                with_bn=False
            )),
            Dense(K * K, K * K, with_bn=False),
            Dense(K * K, K * K, with_bn=False, activation=None)
        )

        self.end_conv = EndChannels(SepConv(
            in_channels=C_mid + C_in,
            out_channels=C_out,
            kernel_size=(1, K),
            depth_multiplier=depth_multiplier
        )).cuda()

    def forward(self, x: Tuple[UFloatTensor,  # (N, P, dims)
                               UFloatTensor,  # (N, P, K, dims)
                               Optional[UFloatTensor]]  # (N, P, K, C_in)
                ) -> UFloatTensor:  # (N, K, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.中心点
          - pts: Regional point cloud.邻居点
          - fts: Regional features.邻居点的特征
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x
        # print('中心点', rep_pt.shape, '。邻居点', pts.shape, '。邻居点特征', fts.shape)
        if fts is not None:
            assert (rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # batch size
            assert (rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # 每个物体点的数量.
            assert (pts.size()[2] == fts.size()[2] == self.K)  # Check K is equal.
            assert (fts.size()[3] == self.C_in)  # Check C_in is equal.
        else:
            assert (rep_pt.size()[0] == pts.size()[0])  # Check N is equal.
            assert (rep_pt.size()[1] == pts.size()[1])  # Check P is equal.
            assert (pts.size()[2] == self.K)  # Check K is equal.
        assert (rep_pt.size()[2] == pts.size()[3] == self.dims)  # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, dims)
        p_center = torch.unsqueeze(rep_pt, dim=2)  # (N, P, 1, dims)

        # 转化为相对中心点的坐标
        pts_local = pts - p_center  # (N, P, K, dims)
        # pts_local = self.pts_layernorm(pts - p_center)

        # 提取邻居点特征(C_mid维)
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted = self.dense2(fts_lifted0)  # (N, P, K, C_mid)

        # 特征拼接
        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.通过mlp得到变换矩阵x
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        # Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim=2)
        return fts_p

class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func: Callable[[UFloatTensor,  # (N, P, dims)
                                           UFloatTensor,  # (N, x, dims)
                                           int, int],
                                          ULongTensor]  # (N, P, K)
                 ) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points. 从pts中选择的中心点
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.
          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in == 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    def select_region(self, pts: UFloatTensor,  # (N, x, dims)
                      pts_idx: ULongTensor  # (N, P, K)
                      ) -> UFloatTensor:  # (P, K, dims)
        """
        根据邻居点的下标得到邻居点
        Selects neighborhood points based on output of r_indices_func.
        :param pts: 原始点云
        :param pts_idx: 邻居点的下标
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([
            pts[n][idx, :] for n, idx in enumerate(torch.unbind(pts_idx, dim=0))
        ], dim=0)
        return regions

    def forward(self, x: Tuple[FloatTensor,  # (N, P, dims)
                               FloatTensor,  # (N, x, dims)
                               FloatTensor]  # (N, x, C_in)
                ) -> FloatTensor:  # (N, P, C_out)
        """
        :param x: (rep_pts, pts, fts) where
          - rep_pts: 从pts中选择的中心点
          - pts: 原始点云
          - fts: pts对应的特征
        :return: Features aggregated to rep_pts.
        """
        rep_pts, pts, fts = x
        # fts = self.dense(fts)

        # KNN on GPU，返回邻居点的下标
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu()).cuda()
        # -------------------------------------------------------------------------- #
        # 根据邻居点的下标得到邻居点和邻居点的特征
        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx)
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))
        # 返回中心点提取特征后的tensor
        return fts_p

class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func: Callable[[UFloatTensor,  # (N, P, dims)
                                           UFloatTensor,  # (N, x, dims)
                                           int, int],
                                          ULongTensor]  # (N, P, K)
                 ) -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x: Tuple[UFloatTensor,
                               UFloatTensor]
                ) -> Tuple[UFloatTensor,
                           UFloatTensor]:
        """
        :param x: (pts, fts) :均为(N, f_dims, num_points)
        where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].  输入的原始点云
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].  pts点云提取的特征
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        pts = pts.permute(0, 2, 1)
        fts = fts.permute(0, 2, 1)
        if 0 < self.P < pts.size()[1]:
            # 选择P个点作为中心点
            idx = np.random.choice(pts.size()[1], self.P, replace=False).tolist()
            rep_pts = pts[:, idx, :]
        else:
            # 全部点都作为中心点
            rep_pts = pts
        # print('中心点',rep_pts.shape,'。全部点',pts.shape,'。全部点特征',fts.shape)
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        # rep_pts_fts:(batch, 特征维度, 点数)
        return rep_pts.permute(0, 2, 1), rep_pts_fts.permute(0, 2, 1)

# c_in:输入特征维度, c_out:输出特征维度, k:邻居点的数量, p:中心点的数量；knn-GPU暂时使用不到d
# return tuple: 第二个元素(batch, 点数, 特征维度)
AbbPointCNN = lambda c_in, c_out, k, d, p: RandPointCNN(c_in, c_out, 3, k, d, p, knn_indices_func_gpu)

class PCNN(nn.Module):
    def __init__(self, n_emb_dims = 512):
        super(PCNN, self).__init__()
        self.n_emb_dims = n_emb_dims

        self.pcnn1 = AbbPointCNN(3, 48, 8, 1, -1)
        self.pcnn2 = AbbPointCNN(48, 96, 8, 2, -1)
        self.pcnn3 = AbbPointCNN(96, 192, 12, 4, -1)
        self.pcnn4 = AbbPointCNN(336, self.n_emb_dims, 16, 4, -1)

    def forward(self, x):
        # input: (batch, 3, num_points)
        x1 = self.pcnn1((x, x))
        x2 = self.pcnn2(x1)
        x3 = self.pcnn3(x2)

        x4 = torch.cat((x1[1], x2[1], x3[1]), dim=1)  # 中心点特征
        out = self.pcnn4((x3[0], x4))[1]
        # return (batch, f_dims, 中心点数)
        return out

# ----- END-PointCNN -----

# ----- PointConv -----
class PointConv(nn.Module):
    def __init__(self, n_emb_dims = 512):
        super(PointConv, self).__init__()
        self.n_emb_dims = n_emb_dims
        self.sa1 = PointConvDensitySetAbstraction(npoint=1024, nsample=32, in_channel=3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=1024, nsample=64, in_channel=128 + 3, mlp=[128, 256, self.n_emb_dims], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1024, nsample=64, in_channel=256 + 3, mlp=[256, 512, self.n_emb_dims], bandwidth = 0.4, group_all=False)

    def forward(self, xyz):
        # (B, 3, num_points)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l2_points

class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        # 计算密度
        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density  # (B, N)

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(xyz, points, inverse_density.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, points, inverse_density.view(B, N, 1))
        # new_xyz: 采样点坐标, [B, npoint, C] (C = 3)
        # new_points: 采样点坐标+特征, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        # 卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        # 密度MLP
        inverse_max_density = grouped_density.max(dim = 2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale
        # 计算权重
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points

def compute_density(xyz, bandwidth):
    '''
    根据欧氏距离计算点云的密度
    xyz: input points position data, [B, N, C]
    '''
    # import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim=-1)
    # return:(B, N)
    return xyz_density

class DensityNet(nn.Module):
    def __init__(self, hidden_unit=[16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale = bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)

        return density_scale

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights = F.relu(bn(conv(weights)))

        return weights

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    # return: (B, S, nsample)
    return group_idx

def sample_and_group(npoint, nsample, xyz, points, density_scale = None):
    """
    采样并且分组(中心点和近邻点作为一组)
    Input:
        npoint: 采样的点数
        nsample: 近邻点数
        xyz: 坐标, [B, N, C] (C = 3)
        points: 点的特征[3:], [B, N, D]
        density_scale: 密度的倒数[B, N, 1]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # 最远点采样，返回采样点下标[B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # 根据下标得到点[B, npoint, C]
    idx = knn_point(nsample, xyz, new_xyz)  # new_xyz作为中心点，最近邻的nsample个点的下标[B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # 分组：中心点和近邻点作为一组.[B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # 坐标归一化，以中心点为原点
    if points is not None:
        # 坐标+特征
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        # 采样点坐标，分组坐标归一化之后的点(坐标+特征)，分组坐标归一化之后的点坐标，中心点和近邻点的下标，分组点的密度
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def sample_and_group_all(xyz, points, density_scale = None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    #new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim = 1, keepdim = True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density

def square_distance(src, dst):
    """
    计算两个点云的欧氏距离
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    根据下标采样点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    最远点采样
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: 采样点的下标 [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    farthest = torch.zeros(B, dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



# -----Transformer -----
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.emb_dims = 512
        self.N = 1
        self.dropout = 0
        self.ff_dims = 1024
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


# torch.cuda.set_device(1)
# src = torch.rand(2, 3, 100).cuda()
# tar = torch.rand(2, 3, 100).cuda()
# net = AGCNN().cuda()
#
# y = net(src)

# print(y.shape)
