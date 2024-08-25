#dgcnn的theta函数
#全局特征和局部特征的融合


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
import transforms3d
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils import ModelNet40_Reg
from model import OverlapNet,CorrNet,mask_point
from scipy.spatial.transform import Rotation
from evaluate_funcs import compute_error, calculate_R_msemae, calculate_t_msemae, evaluate_mask
from loss_util import ChamferLoss
import pointnet2_ops._ext as _ext
from knn_cuda import KNN
from utils import SVD,compute_rigid_transformation,get_keypoints,batch_transform,get_angle_deviation,vis_pc_color
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pointnet2_ops import pointnet2_utils
import open3d as o3d
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0")
torch.cuda.set_device(0)
gpus = [0]
if not os.path.isdir("./logs"):
    os.mkdir("./logs")
writer = SummaryWriter('./logs')
batchsize = 1#16 4
epochs = 400
lr = 1e-3
num_subsampled_rate = 0.5
unseen = True
# unseen = False
noise = False
# file_type = 'modelnet40'
epochs_ = 0
file_type = 'Kitti_odo'
# file_type = 'bunny'
# file_type = '3DMatch_all'

# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(1234)

def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte
def test_one_epoch(net, test_loader, return_t=False):
    net.eval()
    total_loss = 0    
    num_examples = 0
    # loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.HuberLoss(delta=0.4)
    loss_fn = nn.MSELoss()
    cd_loss = ChamferLoss()
    path_res = './results/img-test'
    if not os.path.exists(path_res):
        os.mkdir(path_res)
    with torch.no_grad():
        accs = []
        preciss = []
        recalls = []
        f1s = []
        recons = []
        rmses = []
        tmses = []
        rmaes = []
        tmaes = []
        rres = []
        rtes = []
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        num_keypoints = 256
        num = 0
        for i,data in enumerate(tqdm(test_loader)):
            src, target, rotation, translation, euler, mask_gt,srcun,targetun = data#
            mask_gt_cp = torch.where(mask_gt == 0.0, -1.0, 1.0)
            if use_cuda:
                src = src.cuda()
                target = target.cuda()
            batch_size, num_dims_src, num_points = src.size()
            batch_size, _, num_points_tgt = target.size()
            num_examples += batch_size
            source = src
            #R_pred, t_pred, mask, mask_idx, threshold = net(src, target)
            p, mask_idx,src,target,est_R,est_t,mask_loss,rank_term,loss_confi,pc_s,color_s,pc_t,color_t,lable_ol = net(src, target,eval=True) #,over_los, mask_idxs, mask_idxts
            # est_R,est_t,loss1,loss2,loss3,mask,mask_idx = net(src, target) 
            # if i==1:
                # print(color.sum(axis=1))
                # color_s = mask_gt[0].unsqueeze(-1).cpu().detach().numpy()
                # vis_pc_color(pc_s,color_s)
                # vis_pc_color(pc_t,color_t)
            # src = mask_point(mask_idx, src)
            # out_b = batch_transform(src.permute(0, 2, 1), est_R, est_t.squeeze()).permute(0, 2, 1) 
            # trans_src = batch_transform(src.permute(0, 2, 1), R_pred, t_pred.squeeze()).permute(0, 2, 1)
                        
            # source = src - torch.mean(src, dim=1, keepdim=True)
            # target = target - torch.mean(target, dim=1, keepdim=True)
            
            src_corr = torch.matmul(target, p.transpose(2, 1).contiguous())
            trans_s = torch.bmm(src, p)     
            # src_corr = torch.matmul(src, p)
            # est_R, est_t = compute_rigid_transformation(src_s, src_corr_s, weights)
            # [B,3,N]
            # svd = SVD(emb_dims=1024,input_shape="bcn")       
            # est_R, est_t = svd(trans_s, target)
            # out_a = trans_s
            out_b = batch_transform(trans_s.permute(0, 2, 1), est_R, est_t).permute(0, 2, 1)
            out_a = batch_transform(src_corr.permute(0, 2, 1), est_R.transpose(2,1).contiguous(), -est_t).permute(0, 2, 1)


            rot_threshold = 45#5
            trans_threshold = 10#0.2
            r_deviation = get_angle_deviation(est_R.detach().cpu(), rotation.detach().cpu())
            translation_errors = np.linalg.norm(est_t.detach().cpu()-translation.detach().cpu(),axis=-1)
            flag_1=r_deviation<rot_threshold
            flag_2=translation_errors<trans_threshold
            correct=(flag_1 & flag_2).sum()
            regis_recall=correct/rotation.shape[0]
            # 评估
            # acc, precis, recall, f1 = evaluate_mask(mask_idx, mask_gt)
            acc, precis, recall, f1 = evaluate_mask(lable_ol, mask_gt)
            accs.append(acc)
            preciss.append(precis)
    
            recalls.append(regis_recall)
            f1s.append(f1)
            rotations_ab.append(rotation.detach().cpu().numpy())
            translations_ab.append(translation.detach().cpu().numpy())
            rotations_ab_pred.append(est_R.detach().cpu().numpy())
            translations_ab_pred.append(est_t.detach().cpu().numpy())
            # print(rotation.size(),est_R.size(),translation.size(),est_t.size()) 
            rre = relative_rotation_error(rotation.detach().cpu(), est_R.detach().cpu())  # (*)
            rte = relative_translation_error(translation.detach().cpu(), est_t.detach().cpu())  # (*)
            rres.append(rre)
            rtes.append(rte)
            
            # if False:
            if rre.min()<8 and rte.min()<1:
                num += 1
                out_s = batch_transform(srcun.permute(0, 2, 1).cuda(), est_R, est_t).permute(0, 2, 1)
                points, pointt = trans_s.permute(0,2,1).detach().cpu().numpy(),targetun.permute(0,2,1).detach().cpu().numpy()
                pointp = out_s.permute(0,2,1).detach().cpu().numpy()
                # pointp = out_s.permute(0,2,1).detach().cpu().numpy()
                for cnt in range(batchsize):
                    fig = plt.figure()
                    # 创建一个三维坐标轴
                    ax = plt.axes(projection='3d')
                    # ax.scatter3D(points[cnt, :, 0]-2, points[cnt, :, 1],points[cnt, :, 2], c='green', alpha=0.5, s=1)
                    ax.scatter3D(pointt[cnt, :, 0], pointt[cnt, :, 1],pointt[cnt, :, 2], c='blue', alpha=0.5, s=2)
                    ax.scatter3D(pointp[cnt, :, 0], pointp[cnt, :, 1],pointp[cnt, :, 2], c='red', alpha=0.5, s=2)
                    # ax.scatter3D(pointp[cnt, :, 0]-2, pointp[cnt, :, 1],pointp[cnt, :, 2], c='red', alpha=0.5, s=2)
                    # x_i = np.concatenate((pointt[cnt, :, 0].reshape(-1,1),points[cnt, :, 0].reshape(-1,1)-2),axis = 1)
                    # y_i = np.concatenate((pointt[cnt, :, 1].reshape(-1,1),points[cnt, :, 1].reshape(-1,1)),axis = 1)
                    # z_i = np.concatenate((pointt[cnt, :, 2].reshape(-1,1),points[cnt, :, 2].reshape(-1,1)),axis = 1)
                    # for x_,y_,z_ in zip(x_i,y_i,z_i):
                    #     p_ = np.random.random()
                    #     if p_ < 0.1 :
                    #         ax.plot(x_,y_,z_,c='black',linewidth=0.5)
                    template_pc = pointt.reshape(-1, 3)
                    source_pc = pointp.reshape(-1, 3)
                    template_ = o3d.geometry.PointCloud()
                    source_ = o3d.geometry.PointCloud()
                    template_.points = o3d.utility.Vector3dVector(template_pc)
                    source_.points = o3d.utility.Vector3dVector(source_pc)
                    template_.paint_uniform_color([1, 0, 0])
                    source_.paint_uniform_color([0, 1, 0])
                    o3d.io.write_point_cloud('./results/img-test/PC-T-{}.xyz'.format(i*batchsize + cnt), template_)
                    o3d.io.write_point_cloud('./results/img-test/PC-S-{}.xyz'.format(i*batchsize + cnt), source_)
                    
                    ax.view_init(45, 45)
                    plt.savefig('./results/img-test/pic-{}.png'.format(i*batchsize + cnt))
                    plt.close()   
                    # str_id = "{:0>5}".format(i)
                    # sub_path_s = str_id + "_s.npz" 
                    # sub_path_t = str_id + "_t.npz"  
                    # np.savez(os.path.join("./results/img-test/",sub_path_s), pointp, "data")
                    # np.savez(os.path.join("./results/img-test/",sub_path_t), pointt, "data")  
 

            # est_R = torch.bmm(R_pred, est_R)
            # est_t = t_pred + est_t
            # est_t = torch.bmm(R_pred, t_pred.unsqueeze(2)).squeeze(2) + est_t
            # src = mask_point(mask_idxs, src)
            # target = mask_point(mask_idxt, target)
            # rec_term, rank_term, mfd_term = _run_loss(src, target, p, out_a, out_b)

            # src_topk_idx, src_keypoints, tgt_keypoints = get_keypoints(src, src_corr, weight, num_keypoints)
            # # spatial consistency loss 
            # idx_tgt_corr = torch.argmax(p, dim=-1).int() # [b, n]
            # identity = torch.eye(num_points_tgt).cuda().unsqueeze(0).repeat(batch_size, 1, 1) # [b, m, m]
            # one_hot_number = pointnet2_utils.gather_operation(identity, idx_tgt_corr) # [b, m, n]
            # src_keypoints_idx = src_topk_idx.repeat(1, num_points_tgt, 1) # [b, m, num_keypoints]
            # keypoints_one_hot = torch.gather(one_hot_number, dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * num_keypoints, num_points_tgt)
            # #[b, m, num_keypoints] - [b, num_keypoints, m] - [b * num_keypoints, m]
            # predicted_keypoints_scores = torch.gather(p.transpose(2, 1), dim = 2, index = src_keypoints_idx).transpose(2,1).reshape(batch_size * num_keypoints, num_points_tgt)
            # loss_scl = (-torch.log(predicted_keypoints_scores + 1e-15) * keypoints_one_hot).sum(1).mean()

            # mask_loss = loss_fn(mask , mask_gt.cuda())
            # I_3 = torch.eye(n=3, device=device)
            # I_3 = I_3.unsqueeze(0).repeat(est_R.shape[0], 1, 1)
            # trans_loss = loss_fn(torch.bmm(rotation.cuda().transpose(2,1),est_R),I_3) + loss_fn(translation.cuda(),est_t)
            regist_loss = loss_fn(target,out_b) + loss_fn(src,out_a)
            
            # mask_gt_s = mask_trans(mask_idx,mask_gt)

            # loss = loss_init + 1*rec_term + 1*rank_term + 1 * consensus_loss#+ torch.abs(over_loss)
            #  + 0.1*mfd_term + 10*rec_term + 100*trans_loss
            loss = 1*rank_term +mask_loss  + 100*regist_loss +loss_confi#+ loss_scl
            # loss = loss1.sum() + loss2.sum() + loss3.sum() + mask_loss
            total_loss += loss.item()
            

        
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        rres = np.concatenate(rres, axis=0)
        rtes = np.concatenate(rtes, axis=0)
        np.save('./results/img-test/rres.npy',rres)
        np.save('./results/img-test/rtes.npy',rtes)
        print(rres.min(),rtes.min())
        rmse,rmae = calculate_R_msemae(rotations_ab,rotations_ab_pred)
        tmse,tmae = calculate_t_msemae(translations_ab,translations_ab_pred)
        # t_reshape,t_pred_reshape = translations_ab.reshape(-1,1,3),translations_ab_pred.reshape(-1,1,3)
        # print(t_reshape.shape)
        R_error,t_error = compute_error(rotations_ab,rotations_ab_pred,translations_ab,translations_ab_pred)
        # recon = np.mean(recons)
        acc = np.mean(accs)
        precis = np.mean(preciss)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
    if return_t:
        return total_loss, acc, precis, recall, f1
    return total_loss *1.0/num_examples, acc, np.sqrt(rmse), np.sqrt(tmse), rmae, tmae, R_error,t_error,recall,rres.mean(),rtes.mean(),num




if __name__ == '__main__':

    best_loss = np.inf
    best_acc = 0
    best_rmse = np.inf
    best_recons = np.inf

    # train_loader = DataLoader(
    #     dataset=ModelNet40_Reg(partition='train', max_angle=45, max_t=0.5, unseen=unseen, file_type=file_type,
    #                            num_subsampled_rate=num_subsampled_rate, partial_source=True, noise=noise),
    #     batch_size=batchsize,
    #     shuffle=True,
    #     num_workers=4,
    #     # drop_last=True
    # )
    test_loader = DataLoader(
        dataset=ModelNet40_Reg(partition='test', max_angle=45, max_t=0.5,  unseen=unseen, file_type=file_type,
                               num_subsampled_rate=num_subsampled_rate, partial_source=True, noise=noise),
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    if file_type == 'modelnet40':
        num_subsampled_points = int(num_subsampled_rate * 1024)
    elif file_type in ['3DMatch','3DMatch_all', 'Apollo', 'bunny', 'Kitti_odo']:
        num_subsampled_points = int(num_subsampled_rate * 4096)

    path_res = './results'
    if not os.path.exists(path_res):
        os.mkdir(path_res)

    net = CorrNet(input_pts=num_subsampled_points)
    # net = UNNet(input_pts=num_subsampled_points)
    opt = optim.AdamW(params=net.parameters(), lr=lr)
    # 动态调整学习率
    # scheduler = MultiStepLR(opt, milestones=[50, 100, 150], gamma=0.1)
    if use_cuda:
        net = net.cuda()
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net, device_ids = gpus)

    net.to(device)

    pth = "unseen"#thr-0.9-test-focal_loss-minus-un-k-10-alpha-1
    path_checkpoint = "./checkpoint/%s/ckpt_best_rmse%s.pth"%(pth,str(file_type)+str(num_subsampled_points))  # 断点路径
    # path_checkpoint = "./checkpoint/unseen/ckpt_best_rmse%s.pth"%(str(file_type)+str(num_subsampled_points))  # 断点路径
    checkpoint = torch.load(path_checkpoint,map_location="cuda:0")  # 加载断点
    net.load_state_dict(checkpoint['net'],strict=False)  # 加载模型可学习参数
    
    test_loss, test_acc, test_rmse, test_tmse, test_rmae, test_tmae, R_error, t_error, recall, rre, rte,num = test_one_epoch(net, test_loader)
    print('Test: Loss: %f, Acc: %f, RMSE(R): %f, RMSE(t): %f, MAE(R): %f, MAE(t): %f, R_error: %f, t_error: %f, recall: %f, rre: %f, rte: %f, num:%d'
                    % (test_loss, test_acc, test_rmse, test_tmse, test_rmae, test_tmae,R_error,t_error,recall,rre,rte,num))





#
