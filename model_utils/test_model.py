

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tensorboardX import SummaryWriter
import transforms3d
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils import ModelNet40_Reg
from model import PEAL
from evaluate_funcs import compute_error, calculate_R_msemae, calculate_t_msemae, evaluate_mask
from loss_util import ChamferLoss
from utils import batch_transform,get_angle_deviation
use_cuda = torch.cuda.is_available()
device = 0
torch.cuda.set_device(device)
if not os.path.isdir("./logs"):
    os.mkdir("./logs")
writer = SummaryWriter('./logs')
batchsize = 16#16 4
epochs = 400
lr = 1e-3
num_subsampled_rate = 0.5
unseen = True
# unseen = False
noise = False
# noise = True
file_type = 'modelnet40'
save_path = 'modelnet'
epochs_ = 0
# file_type = 'Kitti_odo'
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
        cds = []
        rres = []
        rtes = []
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        num_keypoints = 256
        num = 0
        for i,data in enumerate(tqdm(test_loader)):
            src, target, rotation, translation, euler, mask_gt = data#
            mask_gt_cp = torch.where(mask_gt == 0.0, -1.0, 1.0)
            if use_cuda:
                src = src.cuda()
                target = target.cuda()
            batch_size, num_dims_src, num_points = src.size()
            batch_size, _, num_points_tgt = target.size()
            num_examples += batch_size
            source = src
            #R_pred, t_pred, mask, mask_idx, threshold = net(src, target)
            p, mask_idx,src,target,est_R,est_t,mask_loss,rank_term,loss_confi,lable_ol = net(src, target,eval=True) #,over_los, mask_idxs, mask_idxts
            src_corr = torch.matmul(target, p.transpose(2, 1).contiguous())
            trans_s = torch.bmm(src, p)     
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
            acc, precis, recall, f1 = evaluate_mask(lable_ol, mask_gt)
            accs.append(acc)
            preciss.append(precis)
    
            recalls.append(regis_recall)
            f1s.append(f1)
            rotations_ab.append(rotation.detach().cpu().numpy())
            translations_ab.append(translation.detach().cpu().numpy())
            rotations_ab_pred.append(est_R.detach().cpu().numpy())
            translations_ab_pred.append(est_t.detach().cpu().numpy())
            rre = relative_rotation_error(rotation.detach().cpu(), est_R.detach().cpu())  # (*)
            rte = relative_translation_error(translation.detach().cpu(), est_t.detach().cpu())  # (*)
            rres.append(rre)
            rtes.append(rte)
            
        
            regist_loss = loss_fn(target,out_b) + loss_fn(src,out_a)
            
            loss = 1*rank_term +mask_loss  + 100*regist_loss +loss_confi
            total_loss += loss.item()
            

        
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        rres = np.concatenate(rres, axis=0)
        rtes = np.concatenate(rtes, axis=0)
        rmse,rmae = calculate_R_msemae(rotations_ab,rotations_ab_pred)
        tmse,tmae = calculate_t_msemae(translations_ab,translations_ab_pred)
        R_error,t_error = compute_error(rotations_ab,rotations_ab_pred,translations_ab,translations_ab_pred)
        acc = np.mean(accs)
        precis = np.mean(preciss)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
    if return_t:
        return total_loss, acc, precis, recall, f1
    return total_loss *1.0/num_examples, acc, np.sqrt(rmse), np.sqrt(tmse), rmae, tmae, R_error,t_error,recall,rres.mean(),rtes.mean()#,np.mean(cds)




if __name__ == '__main__':

    best_loss = np.inf
    best_acc = 0
    best_rmse = np.inf
    best_recons = np.inf

    test_loader = DataLoader(
        dataset=ModelNet40_Reg(partition='test', max_angle=45, max_t=0.5, unseen=unseen,
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

    net = PEAL(input_pts=num_subsampled_points)
    opt = optim.AdamW(params=net.parameters(), lr=lr)
    if use_cuda:
        net = net.cuda()

    net.to(device)
    path_checkpoint = "./checkpoint/%s/ckpt_best_rmse%s.pth"%(save_path,str(file_type)+str(num_subsampled_points))  # 断点路径
    # path_checkpoint = "./checkpoint/unseen/ckpt_best_rmse%s.pth"%(str(file_type)+str(num_subsampled_points))  # 断点路径
    checkpoint = torch.load(path_checkpoint,map_location='cuda:%s'%(str(device)))  # 加载断点
    net.load_state_dict(checkpoint['net'],strict=False)  # 加载模型可学习参数
    
    test_loss, test_acc, test_rmse, test_tmse, test_rmae, test_tmae, R_error, t_error, recall, rre, rte = test_one_epoch(net, test_loader)
    print('Test: Loss: %f, Acc: %f, RMSE(R): %f, RMSE(t): %f, MAE(R): %f, MAE(t): %f, R_error: %f, t_error: %f, recall: %f, rre: %f, rte: %f'
                    % (test_loss, test_acc, test_rmse, test_tmse, test_rmae, test_tmae,R_error,t_error,recall,rre,rte))





#
