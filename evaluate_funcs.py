import torch
import math
import numpy as np
import transforms3d
from scipy.spatial.transform import Rotation
import pytorch3d.transforms as trans3d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_R_msemae(r1, r2, seq='zyx', degrees=True):
    '''
    Calculate mse, mae euler angle error.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    if isinstance(r1, torch.Tensor):
        r1 = r1.cpu().detach().numpy()
    if isinstance(r2, torch.Tensor):
        r2 = r2.cpu().detach().numpy()
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2)**2, axis=-1)
    r_mae = np.mean(np.abs(eulers1 - eulers2), axis=-1)

    return np.mean(r_mse), np.mean(r_mae)


def calculate_t_msemae(t1, t2):
    '''
    calculate translation mse and mae error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    t_mae = np.mean(np.abs(t1 - t2), axis=1)
    return np.mean(t_mse), np.mean(t_mae)


def find_errors(gt_R, pred_R, gt_t, pred_t):
    # gt_R:				ground truth Rotation matrix [3, 3]
    # pred_R: 			predicted rotation matrix [3, 3]
    # gt_t:				ground truth translation vector [1, 3]
    # pred_t: 			predicted translation matrix [1, 3]

    translation_error = np.sqrt(np.sum(np.square(gt_t - pred_t)))
    # Convert matrix remains to axis angle representation and report the angle as rotation error.
    error_mat = np.dot(gt_R.T, pred_R)  # matrix remains [3, 3]=
    rad = np.arccos(np.clip((error_mat - 1) / 2, -1, 1))  # 返回弧度
    angle = abs(rad*(180/np.pi))
    print(angle)
    # 另一种方法计算角度误差
    # rad1 = transforms3d.axangles.mat2axangle(pred_R)[1]
    # rad2 = transforms3d.axangles.mat2axangle(gt_R)[1]
    # angle_our = abs(rad1*(180/np.pi) - rad2*(180/np.pi)) % 360
    return translation_error, angle

def Error_R(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.transpose(0, 2, 1)
    r1r2 = np.matmul(r2_inv, r1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def Error_t(t1, t2, r2):
    '''
    Calculate isotropic translation error between t1 and t2.
    :param t1: shape=(B, 3), pred_t
    :param t2: shape=(B, 3), gtt
    :param R2: shape=(B, 3, 3), gtR
    :return:
    '''
    r2 = r2.transpose(0, 2, 1)
    t2 = np.squeeze(- r2 @ t2[..., None], axis=-1)
    error_t = np.squeeze(r2 @ t1[..., None], axis=-1) + t2
    error_t = np.linalg.norm(error_t, axis=-1)
    return error_t

def compute_error(rotation, rotation_pred, translation, translation_pred):
    # 输入batch个数据
    errors = []
    #计算旋转角度误差和平移误差
    if isinstance(rotation, torch.Tensor):
        rotation = rotation.cpu().detach()
    if isinstance(rotation_pred, torch.Tensor):
        rotation_pred = rotation_pred.cpu().detach()
    if isinstance(translation, torch.Tensor):
        translation = translation.cpu().detach()
    if isinstance(translation_pred, torch.Tensor):
        translation_pred = translation_pred.cpu().detach()

    R_error = np.mean(Error_R(rotation,rotation_pred))
    t_error = np.mean(Error_t(translation,translation_pred,rotation_pred))
    # for gt_R_i, pred_R_i, gt_t_i, pred_t_i in \
    #         zip(rotation, rotation_pred, translation, translation_pred):
    #     errors.append(find_errors(gt_R_i, pred_R_i, gt_t_i, pred_t_i))
    # (t_error, angle_error, angle2_error)仅三个数
    return R_error,t_error


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)

def compute_batch_error(gt_R, pred_R, gt_t, pred_t):
    if isinstance(pred_R, np.ndarray):
        pred_R = torch.from_numpy(pred_R)
    if isinstance(pred_t, np.ndarray):
        pred_t = torch.from_numpy(pred_t)
    if isinstance(gt_R, np.ndarray):
        gt_R = torch.from_numpy(gt_R)
    if isinstance(gt_t, np.ndarray):
        gt_t = torch.from_numpy(gt_t)
    # Tensor: batch个数据
    t_error = torch.mean(torch.norm(gt_t - pred_t, dim=1))

    error_mat = torch.bmm(gt_R.permute(0, 2, 1), pred_R)  # matrix remains [batch, 3, 3]
    rad = trans3d.so3_rotation_angle(error_mat)
    R_error = torch.mean(abs(rad * (180 / np.pi)))

    rad1 = trans3d.so3_rotation_angle(pred_R)
    rad2 = trans3d.so3_rotation_angle(gt_R)
    R_error2 = torch.mean(abs(rad1 * (180 / np.pi) - rad2 * (180 / np.pi)) % 360)
    return [t_error, R_error, R_error2]


def evaluate_mask(mask, mask_gt):
    accs = []
    preciss = []
    recalls = []
    f1s = []
    for m, m_gt in zip(mask, mask_gt):
        m = m.cpu()
        m_gt = m_gt.cpu()
        # mask, mask_gt: n维
        acc = accuracy_score(m_gt, m.detach())
        precis = precision_score(m_gt, m.detach(), zero_division=0)
        recall = recall_score(m_gt, m.detach(), zero_division=0)
        f1 = f1_score(m_gt, m.detach())

        accs.append(acc)
        preciss.append(precis)
        recalls.append(recall)
        f1s.append(f1)
    acc = np.mean(accs)
    precis = np.mean(preciss)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    return acc, precis, recall, f1


