import os
import random
from typing import Any, Dict, List, Optional
import matplotlib as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose
from libs.models.graph.graph import Graph
from libs.models.graph.tools import k_adjacency, normalize_adjacency_matrix, get_adjacency_matrix
from utils.makeGrid import migrate_batch_points

__all__ = ["ActionSegmentationDataset", "collate_fn"]

dataset_names = ["MCFS-130", "PKU-subject", "PKU-view", "LARA", "TCG-15"]
modes = ["training", "validation", "trainval", "test"]

def get_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    t = sample.shape[1]
    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    # Shape: C, T, V, M
    final_sample[:, start:end-1, :, :] = disps

    return final_sample

def get_relative_coordinates(sample,
                             references=(0)):
    # input: C, T, V, M
    # references=(4, 8, 12, 16)
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    C, t, V, M = sample.shape
    rel_coords = []
    #for i in range(len(references)):
    ref_loc = sample[:, :, references, :]
    coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
    rel_coords.append(coords_diff)
    
    # Shape: C, t, V, M 
    rel_coords = np.vstack(rel_coords)
    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_coords
    return final_sample

def norm_feature(feature):
    C, T, V, M= feature.shape

    min_vals = np.min(feature, axis=(1, 2, 3), keepdims=True)
    max_vals = np.max(feature, axis=(1, 2, 3), keepdims=True)

    range_vals = max_vals - min_vals
    epsilon = 1e-8

    range_vals = np.where(range_vals == 0, epsilon, range_vals)
    non_zero = range_vals != 0

    norm_f = np.where(non_zero, (feature - min_vals) / range_vals, 0)

    return norm_f

class ActionSegmentationDataset(Dataset):
    """ Action Segmentation Dataset """

    def __init__(
        self,
        dataset: str,
        transform: Optional[Compose] = None,
        mode: str = "training",
        split: int = 1,
        dataset_dir: str = "../dataset",
        csv_dir: str = "./csv",
        augmentations = True,
    ) -> None:
        super().__init__()
        """
            Args:
                dataset: the name of dataset
                transform: torchvision.transforms.Compose([...])
                mode: training, validation, test
                split: which split of train, val and test do you want to use in csv files.(default:1)
                csv_dir: the path to the directory where the csv files are saved
        """
        self.mode = mode
        self.augmentations = augmentations
        assert (
            dataset in dataset_names
        ), "You have to choose dataset."

        if mode == "training":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
        elif mode == "validation":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "val{}.csv".format(split))
            )
        elif mode == "trainval":
            df1 = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
            df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
            self.df = pd.concat([df1, df2])
        elif mode == "test":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "test{}.csv".format(split))
            )
        else:
            assert (
                mode in modes
            ), "You have to choose 'training', 'trainval', 'validation' or 'test' as the dataset mode."

        self.transform = transform
        self.dataset = dataset

        self.graph = Graph(labeling_mode='spatial', layout=dataset)
        neighbor = self.graph.neighbor

        if dataset == 'LARA':
            self.A = get_adjacency_matrix(neighbor, 19)
        elif dataset == 'TCG-15':
            self.A = get_adjacency_matrix(neighbor, 17)
        else:
            self.A = get_adjacency_matrix(neighbor, 25)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature_path = self.df.iloc[idx]["feature"]
        label_path = self.df.iloc[idx]["label"]
        boundary_path = self.df.iloc[idx]["boundary"]

        feature = np.load(feature_path, allow_pickle=True).astype(np.float32)

        if (self.dataset == 'MCFS-22') or (self.dataset == 'MCFS-130'):
            feature = feature[:, :, :2]  # t,v,c
            feature[:, :, 0] = feature[:, :, 0] / 1280 - 0.5
            feature[:, :, 1] = feature[:, :, 1] / 720 - 0.5
            feature = feature - feature[:, 8:9, :]
            feature = feature.transpose(2, 1, 0)  # t,v,c--->c,v,t

            motion = np.concatenate((feature[:, :, 0:1], feature[:, :, 1:] - feature[:, :, :-1]), axis=-1)
            # motion = fft_diff(feature)
            feature = np.concatenate([motion, feature], axis=0)

        elif (self.dataset == 'PKU-subject') or (self.dataset == 'PKU-view'):
            feature = feature.reshape(-1,2,25,3).transpose(3,0,2,1) #   t,m,v,c--->c,t,v,m
            # c, t, v, m = feature.shape
            # feature = feature[:,:,:,0].reshape(c, t, v, 1)
            # if np.all(temp==0):
            #     print("all is zero")
            # else:
            #     print("not all zero")
            feature = norm_feature(feature)
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3,0,2,1).reshape(12, 25, -1) #   c,t,v,m--->mc,v,t

            motion = np.concatenate((feature[:, :, 0:1], feature[:, :, 1:] - feature[:, :, :-1]), axis=-1)
            feature = np.concatenate([motion, feature], axis=0)
        
        elif  (self.dataset == 'LARA'):
            feature = norm_feature(feature)
            disps = get_displacements(feature).transpose(3,0,2,1).reshape(6, 19, -1)
            # sqrt_difference = sqrt_diff(feature.transpose(3,0,2,1).reshape(6,19,-1))
            # norm_difference = norm_diff(feature.transpose(3, 0, 2, 1).reshape(6, 19, -1))
            rel_coords = get_relative_coordinates(feature).transpose(3,0,2,1).reshape(6, 19, -1)
            # feature = np.concatenate([disps, rel_coords], axis=0)
            # feature = feature.transpose(3,0,2,1).reshape(12, 19, -1) #   c,t,v,m--->mc,v,t
            feature = np.concatenate([disps, rel_coords], axis=0)

        elif (self.dataset == 'TCG-15'):
            feature = feature.transpose(2,0,1)[...,np.newaxis]
            # feature = norm_feature(feature)
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3,0,2,1).reshape(6,17,-1)

            motion = np.concatenate((feature[:, :, 0:1], feature[:, :, 1:] - feature[:, :, :-1]), axis=-1)
            feature = np.concatenate([motion, feature], axis=0)

        # img = skeleton_img_xy.transpose(1,2,0)
        label = np.load(label_path).astype(np.int64)
        boundary = np.load(boundary_path).astype(np.float32)
        
            
        if self.transform is not None:
            feature, label, boundary = self.transform([feature, label, boundary])

        if self.mode != "test":
            if self.augmentations:
                C, V, T = feature.shape

                random_index = torch.rand(1).item()

                # 关节遮挡
                if random_index < 0.35:
                    masked_data = feature.clone()
                    # Randomly determine the number of joints to mask
                    num_masked_joints = torch.randint(1, int(V * 0.5) + 1, (1,)).item()
                    # Randomly select the indices of the joints to mask
                    masked_joint_indices = torch.randperm(V)[:num_masked_joints]
                    # Mask the selected joints across all frames and channels
                    masked_data[:, masked_joint_indices, :] = 0
                    feature = masked_data

                # 整体旋转
                if random_index > 0.65:
                    if (self.dataset == 'MCFS-22') or (self.dataset == 'MCFS-130'):
                        feature = feature
                        # feature = torch.cat((feature[:C//2, :, :], torch.zeros(1, V, T), feature[C//2:, :, :], torch.zeros(1, V, T)), dim=0)
                        # feature = random_rot_3axis(feature, self.dataset)
                        # feature = torch.cat((feature[:C//2, :, :], feature[C//2+1:C//2+3, :, :]), dim=0)

                    else:
                        feature = random_rot_3axis(feature, self.dataset)  # 作一个随机的旋转变换

        sample = {
            "feature": feature,
            "label": label,
            "feature_path": feature_path,
            "boundary": boundary,
        }

        return sample

def random_rot_3axis(data_torch, dataset, theta= torch.pi):
    """
    data_numpy: C,V,T
    """
    C, V, T = data_torch.shape #torch.Size([3, 64, 25, 2])
    data_torch = data_torch.contiguous().view(-1, 3, V, T)  # T,3,V*M
    # rot = torch.zeros(3).uniform_(-theta, theta) #初始化旋转角度 即每帧的row，pitch，yaw相同
    # 创建一个包含三个元素的零张量
    if dataset == "LARA" or dataset == "TCG-15":
        rot = torch.zeros(3)
        rot[2] = torch.FloatTensor(1).uniform_(-theta, theta)
        # rot[:2] = torch.FloatTensor(2).uniform_(-0.3, 0.3)
    elif (dataset == 'PKU-subject') or (dataset == 'PKU-view'):
        rot = torch.zeros(3)
        rot[1] = torch.FloatTensor(1).uniform_(-theta, theta)
        # rot[0] = torch.FloatTensor(1).uniform_(-0.3, 0.3)
        # rot[2] = torch.FloatTensor(1).uniform_(-0.3, 0.3)
    elif (dataset == 'MCFS-22') or (dataset == 'MCFS-130'):
        rot = torch.zeros(3)
        rot[1] = torch.FloatTensor(1).uniform_(-theta, theta)
        # rot[0] = torch.FloatTensor(1).uniform_(-0.3, 0.3)
        # rot[2] = torch.FloatTensor(1).uniform_(-0.3, 0.3)
    rot = _rot_3axis(rot)  # T,3,3  #直接转换成定轴旋转矩阵
    data_torch = torch.einsum('dc,ncvt->ndvt', rot, data_torch) #旋转变换
    data_torch = data_torch.contiguous().view(C, V, T) #（3,64,25,2）

    return data_torch

def random_rot_2axis(data_torch, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    C, V, T = data_torch.shape #torch.Size([3, 64, 25, 2])
    data_torch = data_torch.contiguous().view(-1, 2, V, T)  # T,3,V*M
    rot = torch.zeros(1).uniform_(-theta, theta) #初始化旋转角度 即每帧的row，pitch，yaw相同
    rot = _rot_2axis(rot)  # T,3,3  #直接转换成定轴旋转矩阵
    data_torch = torch.einsum('dc,ncvt->ndvt', rot, data_torch) #旋转变换
    data_torch = data_torch.contiguous().view(C, V, T) #（3,64,25,2）

    return data_torch

def _rot_3axis(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3# 计算每个维度的旋转角度的余弦和正弦值
    zeros = torch.zeros(1)  # T,1 # 创建全零和全一的张量
    ones = torch.ones(1)  # T,1
    # 构造 X 轴的旋转矩阵
    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[0:1], -sin_r[0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, sin_r[0:1], cos_r[0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 0)  # T,3,3
    # 构造 Y 轴的旋转矩阵
    ry1 = torch.stack((cos_r[1:2], zeros, sin_r[1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((-sin_r[1:2], zeros, cos_r[1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 0)
    # 构造 Z 轴的旋转矩阵
    rz1 = torch.stack((cos_r[2:3], -sin_r[2:3], zeros), dim =-1)
    rz2 = torch.stack((sin_r[2:3], cos_r[2:3], zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz = torch.cat((rz1, rz2, r3), dim = 0)

    rot = rz.matmul(ry).matmul(rx) #左乘我记得是绕定轴转动，所以这个是row，yaw,pitch
    return rot

def _rot_2axis(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3# 计算每个维度的旋转角度的余弦和正弦值
    # 构造 X 轴的旋转矩阵
    rx1 = torch.stack((cos_r, -sin_r), dim = -1)  # T,1,3
    rx2 = torch.stack((sin_r, cos_r), dim = -1)  # T,1,3
    rx = torch.cat((rx1, rx2), dim = 0)  # T,3,3
    # 构造 Y 轴的旋转矩阵

    return rx


def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([s["feature"].shape[2] for s in sample])

    feat_list = []
    label_list = []
    path_list = []
    boundary_list = []
    length_list = []

    for s in sample:
        feature = s["feature"]
        label = s["label"]
        boundary = s["boundary"]
        feature_path = s["feature_path"]

        _, _, t = feature.shape
        pad_t = max_length - t
        length_list.append(t)

        if pad_t > 0:
            feature = F.pad(
                feature, (0, pad_t), mode='constant', value=0.)
            label = F.pad(label, (0, pad_t), mode='constant', value=255)  #255
            boundary = F.pad(boundary, (0, pad_t), mode='constant', value=0.)

        # reshape boundary (T) => (1, T)
        boundary = boundary.unsqueeze(0)

        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)
        boundary_list.append(boundary)

    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)

    # merge labels from tuple of 2D tensor to 3D tensor
    # shape (N, 1, T)
    boundaries = torch.stack(boundary_list, dim=0)

    # generate masks which shows valid length for each video (N, 1, T)
    masks = [
        [[1 if i < length else 0 for i in range(max_length)]] for length in length_list
    ]
    masks = torch.tensor(masks, dtype=torch.bool)

    return {
        "feature": features,
        "label": labels,
        "boundary": boundaries,
        "feature_path": path_list,
        "mask": masks,
    }


def random_occlude_features(feature, value=0):

    C, V, T = feature.shape
    # 计算需要遮挡的特征维度数量
    occlusion_ratio =torch.rand(1) * 0.5

    # 计算遮挡区域的大小
    occlusion_size_V = int(V * occlusion_ratio)
    occlusion_size_T = int(T * occlusion_ratio)

    # 随机选择遮挡的起始点
    start_V = np.random.randint(0, V - occlusion_size_V + 1)
    start_T = np.random.randint(0, T - occlusion_size_T + 1)

    # 创建一个与输入张量形状相同的全零张量
    occluded_tensor = np.zeros_like(feature)

    # 将非遮挡区域的值复制到遮挡张量中
    occluded_tensor[:, start_V:start_V + occlusion_size_V, start_T:start_T + occlusion_size_T] = feature[:,
                                                                                                 start_V:start_V + occlusion_size_V,
                                                                                                 start_T:start_T + occlusion_size_T]

    return feature

def random_rotation_features(feature):
    C, V, T = feature.shape
    if C % 3 == 0:
        """生成一个随机的旋转矩阵"""
        theta = torch.rand(1)* 2 * torch.pi
        random_rot = random.randint(0, 2)
        rot_matrix = [torch.tensor([
            [1, 0, 0],
            [0, torch.cos(theta), -torch.sin(theta)],
            [0, torch.sin(theta), torch.cos(theta)]
        ]), torch.tensor([
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)]
        ]), torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ])]
        rot_m = torch.zeros((C,C))
        for i in range(C//3):
            rot_m[i*3:(i+1)*3,i*3:(i+1)*3] = rot_matrix[random_rot]
    elif C % 2 == 0:
        """生成一个随机的旋转矩阵"""
        theta = torch.rand(1) * 2 * torch.pi
        rot_matrix = torch.tensor(([torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)]))

        rot_m = torch.zeros((C, C))
        for i in range(C // 2):
            rot_m[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = rot_matrix

    feature = torch.einsum('cd, cvt -> dvt', rot_m, feature)

    return feature

def fft_diff(feature):
    C,V,T = feature.shape
    feature = feature.transpose(2,0,1) # T, C, V
    fft_features = np.fft.fft2(feature, axes=(1,2))
    fft_features = np.abs(fft_features - np.roll(fft_features, shift=1, axis=0))

    epsilon = 1e-10
    # 计算差异的幅度谱
    magnitude_spectrum = 20 * np.log(np.abs(fft_features) + epsilon)

    return magnitude_spectrum.transpose(1,2,0)

#pingfangchafen
def sqrt_diff(feature):
    C,V,T = feature.shape
    sqrt_differences = (feature[:,:,:1:] - feature[:,:,:-1]) ** 2
    sqrt_differences = np.concatenate([np.zeros([C, V, 1]), sqrt_differences], axis=-1)

    return sqrt_differences

#guiyihuachafen
def norm_diff(feature):
    C,V,T = feature.shape
    diff = feature[:, :, 1:] - feature[:,:,:-1]
    max_diff = np.max(np.abs(diff), axis=(1,2), keepdims=True)
    norm_differences = diff / max_diff
    norm_differences = np.concatenate([np.zeros([C, V, 1]), norm_differences], axis=-1)

    return norm_differences

