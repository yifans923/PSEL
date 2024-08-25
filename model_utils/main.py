
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
import sys, os
sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from data_utils import ModelNet40_Reg
from model import PEAL
from scipy.spatial.transform import Rotation
from evaluate_funcs import calculate_R_msemae, calculate_t_msemae, evaluate_mask
from utils import batch_transform

use_cuda = torch.cuda.is_available()
device = 0
torch.cuda.set_device(device)
if not os.path.isdir("./logs"):
    os.mkdir("./logs")
writer = SummaryWriter('./logs')
batchsize = 16 # modelnet
epochs = 300
lr = 1e-3
num_subsampled_rate = 0.5
unseen = True
noise = False


save_path = '2024-4-15'
resume = False

epochs_ = 0
file_type = 'modelnet40'
# file_type = 'Kitti_odo'
# file_type = '3DMatch_all'
# file_type = 'bunny'


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


def test_one_epoch(net, test_loader, return_t=False):
    net.eval()
    total_loss = 0
    num_examples = 0
    loss_fn = nn.MSELoss()
    global epochs_

    with torch.no_grad():
        accs,preciss,recalls,f1s,rotations_ab,translations_ab,rotations_ab_pred, translations_ab_pred = [],[],[],[],[],[],[],[]
        for i,data in enumerate(tqdm(test_loader)):
            source, target, rotation, translation, euler, mask_gt = data
            mask_gt_cp = torch.where(mask_gt == 0.0, -1.0, 1.0)
            if use_cuda:
                source = source.cuda()
                target = target.cuda()
            batch_size, num_dims_src, num_points = source.size()
            batch_size, _, num_points_tgt = target.size()
            num_examples += batch_size
            p, mask_idx,src,target,est_R,est_t,mask_loss,rank_term,loss_confi,label_ol = net(source, target) 
            
            src_corr = torch.matmul(target, p.transpose(2, 1).contiguous())
            trans_s = torch.bmm(src, p)     
            out_b = batch_transform(trans_s.permute(0, 2, 1), est_R, est_t.squeeze()).permute(0, 2, 1)
            out_a = batch_transform(src_corr.permute(0, 2, 1), est_R.transpose(2,1).contiguous(), -est_t.squeeze()).permute(0, 2, 1)
            
            regist_loss = loss_fn(target,out_b) + loss_fn(src,out_a)
            
            loss =  1*rank_term + mask_loss  +100*regist_loss+loss_confi
            total_loss += loss.item()
            # 评估
            acc, precis, recall, f1 = evaluate_mask(mask_idx, mask_gt)            
            accs.append(acc)
            preciss.append(precis)
            recalls.append(recall)
            f1s.append(f1)
            rotations_ab.append(rotation.detach().cpu().numpy())
            translations_ab.append(translation.detach().cpu().numpy())
            rotations_ab_pred.append(est_R.detach().cpu().numpy())
            translations_ab_pred.append(est_t.detach().cpu().numpy())
        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        rmse,mae = calculate_R_msemae(rotations_ab,rotations_ab_pred)
        tmse,mae = calculate_t_msemae(translations_ab,translations_ab_pred)
        # recon = np.mean(recons)
        acc = np.mean(accs)
        precis = np.mean(preciss)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
    epochs_ += 1
    if return_t:
        return total_loss, acc, precis, recall, f1

    return total_loss *1.0/num_examples, acc, np.sqrt(rmse), np.sqrt(tmse), rmae, tmae


def train_one_epoch(net, opt, train_loader, return_t=False):
    net.train()
    total_loss = 0
    num_examples = 0
    accs,preciss,recalls,f1s,rotations_ab,translations_ab,rotations_ab_pred, translations_ab_pred = [],[],[],[],[],[],[],[]
    loss_fn = nn.MSELoss()
    for i,data in enumerate(tqdm(train_loader)):
        source, target, rotation, translation, euler, mask_gt = data
        # 用于计算损失的阈值，mask_gt用于计算准确率
        mask_gt_cp = torch.where(mask_gt == 0.0, -1.0, 1.0)
        if use_cuda:
            source = source.cuda()
            target = target.cuda()
        batch_size, num_dims_src, num_points = source.size()
        batch_size, _, num_points_tgt = target.size()
        num_examples += batch_size
        
        opt.zero_grad()
        p, mask_idx,src,target,est_R,est_t,mask_loss,rank_term,loss_confi,label_ol = net(source, target) 
        src_corr = torch.matmul(target, p.transpose(2, 1).contiguous())
        trans_s = torch.bmm(src, p) 
        out_b = batch_transform(trans_s.permute(0, 2, 1), est_R, est_t.squeeze()).permute(0, 2, 1)
        out_a = batch_transform(src_corr.permute(0, 2, 1), est_R.transpose(2,1).contiguous(), -est_t.squeeze()).permute(0, 2, 1)
        regist_loss = loss_fn(target, out_b) + loss_fn(src, out_a)
        loss =  1*rank_term + mask_loss  +100*regist_loss+loss_confi
        total_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5, norm_type=2)
        opt.step()

        # 评估
        acc, precis, recall, f1 = evaluate_mask(label_ol, mask_gt)    
        accs.append(acc)
        preciss.append(precis)
        recalls.append(recall)
        f1s.append(f1)
        #acc, precis, recall, f1 = evaluate_mask(mask_idxs, mask_gt_s)
        rotations_ab.append(rotation.detach().cpu().numpy())
        translations_ab.append(translation.detach().cpu().numpy())
        rotations_ab_pred.append(est_R.detach().cpu().numpy())
        translations_ab_pred.append(est_t.detach().cpu().numpy())
    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
    rmse,rmae = calculate_R_msemae(rotations_ab,rotations_ab_pred)
    tmse,tmae = calculate_t_msemae(translations_ab,translations_ab_pred)
    # recon = np.mean(recons)
    acc = np.mean(accs)
    precis = np.mean(preciss)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    if return_t:
        return total_loss, acc, precis, recall, f1

    return total_loss*1.0/num_examples, acc, np.sqrt(rmse), np.sqrt(tmse), rmae, tmae


if __name__ == '__main__':

    best_loss = np.inf
    best_acc = 0
    best_rmse = np.inf
    best_recons = np.inf

    train_loader = DataLoader(
        dataset=ModelNet40_Reg(partition='train', max_angle=45, max_t=0.5, unseen=unseen, file_type=file_type,
                               num_subsampled_rate=num_subsampled_rate, partial_source=True, noise=noise),
        batch_size=batchsize,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    test_loader = DataLoader(
        dataset=ModelNet40_Reg(partition='test', max_angle=45, max_t=0.5,  unseen=unseen, file_type=file_type,
                               num_subsampled_rate=num_subsampled_rate, partial_source=True, noise=noise),
        batch_size=batchsize,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    if file_type == 'modelnet40':
        num_subsampled_points = int(num_subsampled_rate * 1024)
    elif file_type in ['Apollo', 'bunny','Kitti_odo']:
        num_subsampled_points = int(num_subsampled_rate * 4096)
    elif file_type in ['3DMatch', '3DMatch_all']:
        num_subsampled_points = int(num_subsampled_rate * 4096)
    path_res = './results'
    if not os.path.exists(path_res):
        os.mkdir(path_res)

    net = PEAL(input_pts=num_subsampled_points,filetype=file_type)
    opt = optim.AdamW(params=net.parameters(), lr=lr)
    
    if use_cuda:
        net = net.cuda()

    check_path = save_path
    start_epoch = -1
    RESUME = resume   # 是否加载模型继续上次训练
    if RESUME:
        print('Loading latest model----')
        path_checkpoint = "./checkpoint/%s/ckpt%s.pth"%(str(check_path),str(file_type)+str(num_subsampled_points))  # 断点路径

        checkpoint = torch.load(path_checkpoint,map_location='cuda:%s'%(str(device)))  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        opt.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        # 加载上次best结果
        best_loss = checkpoint['best_loss']
        best_acc = checkpoint['best_acc']
        best_rmse = checkpoint['best_rmse']
        best_tmse = checkpoint['best_tmse']

    accs = []
    rmses = []
    print(num_subsampled_rate)
    for epoch in range(start_epoch + 1, epochs):

        train_loss, train_acc, train_rmse, train_tmse,rmae,tmae = train_one_epoch(net, opt, train_loader)
        test_loss, test_acc, test_rmse, test_tmse,rmae,tmae = test_one_epoch(net, test_loader)

        if test_loss <= best_loss:
            best_loss = test_loss
            best_acc = test_acc
            # 保存最好的checkpoint
            checkpoint_best = {
                "net": net.state_dict(),
            }
            if not os.path.isdir("./checkpoint/%s"%(str(check_path))):
                os.mkdir("./checkpoint/%s"%(str(check_path)))
            torch.save(checkpoint_best, './checkpoint/%s/ckpt_best%s.pth'%(str(check_path),str(file_type)+str(num_subsampled_points)))
        if test_rmse <= best_rmse:
            best_rmse = test_rmse
            best_tmse = test_tmse
            # 保存最好的checkpoint
            checkpoint_best = {
                "net": net.state_dict(),
            }
            if not os.path.isdir("./checkpoint/%s"%(str(check_path))):
                os.mkdir("./checkpoint/%s"%(str(check_path)))
            torch.save(checkpoint_best, './checkpoint/%s/ckpt_best_rmse%s.pth'%(str(check_path),str(file_type)+str(num_subsampled_points)))
        print('---------Epoch: %d---------' % (epoch+1))
        print('Train: Loss: %f, Acc: %f, RMSE(R): %f, RMSE(t): %f'
                      % (train_loss, train_acc, train_rmse, train_tmse))

        print('Test: Loss: %f, Acc: %f, RMSE(R): %f, RMSE(t): %f'
                      % (test_loss, test_acc, test_rmse, test_tmse))

        print('Best: Loss: %f, Acc: %f, RMSE(R): %f, RMSE(t): %f'
                      % (best_loss, best_acc, best_rmse, best_tmse))
        writer.add_scalar('Train/train loss', train_loss, global_step=epoch)
        writer.add_scalar('Train/train Acc', train_acc, global_step=epoch)
        writer.add_scalar('Train/train Rmse', train_rmse, global_step=epoch)
        writer.add_scalar('Train/train tmse', train_tmse, global_step=epoch)

        writer.add_scalar('Test/test loss', test_loss, global_step=epoch)
        writer.add_scalar('Test/test Acc', test_acc, global_step=epoch)
        writer.add_scalar('Test/test Rmse', test_rmse, global_step=epoch)
        writer.add_scalar('Test/test tmse', test_tmse, global_step=epoch)

        writer.add_scalar('Best/best loss', best_loss, global_step=epoch)
        writer.add_scalar('Best/best Acc', best_acc, global_step=epoch)
        writer.add_scalar('Best/best Rmse', best_rmse, global_step=epoch)
        writer.add_scalar('Best/best tmse', best_tmse, global_step=epoch)

        # 保存checkpoint
        checkpoint = {
            "net": net.state_dict(),
            'optimizer': opt.state_dict(),
            "epoch": epoch,
            "best_loss":best_loss,
            'best_acc': best_acc,
            'best_rmse': best_rmse,
            'best_tmse': best_tmse, 
        }
        if not os.path.isdir("./checkpoint/%s"%(str(check_path))):
            os.mkdir("./checkpoint/%s"%(str(check_path)))
        torch.save(checkpoint, './checkpoint/%s/ckpt%s.pth'%(str(check_path),str(file_type)+str(num_subsampled_points)))
    writer.close()


#
