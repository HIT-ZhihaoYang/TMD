import colorsys
import os
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math
import numpy as np
from libs.class_id_map import get_id2class_map, get_n_classes
from libs.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from libs.postprocess import PostProcessor
from libs.tools import segment_video_labels, gen_label, generate_segment_features, create_logits, gen_transition_label, generate_transition_features
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def split_and_pad_tensor(tensor1, B=128, return_pm=False):
    # 获取张量的维度
    N, H, W, T = tensor1.shape

    num_splits = T // B  # 计算可以整除的分割数量
    remainder = T % B  # 计算余数

    if remainder != 0:
        splits1 = torch.cat((tensor1, torch.zeros((N,H,W,int(B*(num_splits+1)-T))).to(tensor1.device)),dim=3)
        splits1= splits1.view(N, H, W, num_splits+1, B)
        result1 = splits1.permute(3, 0, 1, 2, 4)

        return result1, int(B * (num_splits + 1) - T)

    else:
        splits1 = tensor1.view(N, H, W, num_splits, B)
        result1 = splits1.permute(3, 0, 1, 2, 4)

        return result1 , 0

def train(
    train_loader: DataLoader,
    model: nn.Module,
    action_embeddings_data,
    action_embeddings_graph,
    joint_embeddings_graph,
    motion_embeddings_data,
    maintain_embeddings_data,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
criterion_contrast: nn.Module,
    lambda_bound_loss: float,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    losses = AverageMeter("Loss", ":.4e")

    # switch training mode
    model.train()
    for sample in train_loader:
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]

        x = x.to(device)
        t = t.to(device)
        b = b.to(device)
        mask = mask.to(device)

        batch_size = x.shape[1]

        # pure的对比学习
        t_segment = segment_video_labels(t)

        #find all labels(segments) in this batch
        label = [i[0] for seg in t_segment for i in seg]
        transition_label = gen_transition_label(t_segment)

        label_g = gen_label(label)  # （N，N）对比矩阵的GT

        transition_label_g = gen_label(transition_label)  # （N，N）对比矩阵的GT


        # compute output and loss
        output_cls, output_bound, output_feature, output_motion, logit_scale = model(x, mask,
                                                                                                   joint_embeddings_graph)  # 模型计算边界以及计算分类结果

        action_embedding = list()
        for single_label in label:  # n的list里面都是（1,77）的对应的句索引
            action_item = action_embeddings_data[single_label].unsqueeze(dim=0)  # 拿到这个类的任意一个同义句的index
            action_embedding.append(action_item)

        action_embedding = torch.cat(action_embedding).cuda(device)

        action_features = []
        if isinstance(output_feature, list):
            for i in range(len(output_feature)):
                action_feature = generate_segment_features(output_feature[i], t_segment, device)
                action_features.append(action_feature)

        # motion embedding
        motion_embedding = list()
        for single_label in transition_label:  # n的list里面都是（1,77）的对应的句索引
            motion_item = motion_embeddings_data[single_label[0], single_label[1], :].unsqueeze(
                dim=0)  # 拿到这个类的任意一个同义句的index
            motion_embedding.append(motion_item)

        motion_embedding = torch.cat(motion_embedding).cuda(device)
        motion_embedding = motion_embedding.float()

        maintain_embedding = list()
        for single_label in label:  # n的list里面都是（1,77）的对应的句索引
            maintain_item = maintain_embeddings_data[single_label].unsqueeze(dim=0)  # 拿到这个类的任意一个同义句的index
            maintain_embedding.append(maintain_item)

        maintain_embedding = torch.cat(maintain_embedding).cuda(device)
        maintain_embedding = maintain_embedding.float()

        motion_features = []
        maintain_features = []
        if isinstance(motion_features, list):
            # for i in range(len(output_feature)):
            motion_feature, maintain_feature = generate_transition_features(output_motion[0], t_segment, device)
            motion_features.append(motion_feature)
            maintain_features.append(maintain_feature)

        loss = 0.0
        if isinstance(output_cls, list):
            n = len(output_cls)
            for out in output_cls:
                loss += criterion_cls(out, t, x) / n
        else:
            loss += criterion_cls(output_cls, t, x)

        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

        if isinstance(action_features, list):
            for i in range(len(action_features)):
                logits_per_image, logits_per_text = create_logits(action_features[i], action_embedding, logit_scale[0])  # 计算两个cosine相似度
                ground_truth = torch.tensor(label_g, dtype=action_features[0].dtype, device=device)  # GT矩阵变成对应格式以及设备

                loss_imgs = criterion_contrast(logits_per_image, ground_truth)  # 计算对比损失KLL
                loss_texts = criterion_contrast(logits_per_text, ground_truth)

                loss += 0.8 * ((loss_imgs + loss_texts) / 2)  # 二者损失平均放入list

        if isinstance(maintain_features, list):
            logits_per_image, logits_per_text = create_logits(maintain_features[0], maintain_embedding, logit_scale[1])  # 计算两个cosine相似度
            ground_truth = torch.tensor(label_g, dtype=maintain_features[0].dtype, device=device)  # GT矩阵变成对应格式以及设备

            loss_imgs = criterion_contrast(logits_per_image, ground_truth)  # 计算对比损失KLL
            loss_texts = criterion_contrast(logits_per_text, ground_truth)
            loss_imgs_texts = loss_imgs + loss_texts
            if ~torch.isnan(loss_imgs_texts):
                loss += 0.5 * ((loss_imgs + loss_texts) / 2)  # 二者损失平均放入list
        
        if isinstance(motion_features, list):
            logits_per_image, logits_per_text = create_logits(motion_features[0], motion_embedding, logit_scale[1])  # 计算两个cosine相似度
            transition_ground_truth = torch.tensor(transition_label_g, dtype=motion_features[0].dtype, device=device)  # GT矩阵变成对应格式以及设备

            loss_imgs = criterion_contrast(logits_per_image, transition_ground_truth)  # 计算对比损失KLL
            loss_texts = criterion_contrast(logits_per_text, transition_ground_truth)

            loss += 0.5 * ((loss_imgs + loss_texts) / 2)  # 二者损失平均放入list

        # record loss
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(
    val_loader: DataLoader,
    model: nn.Module,
    joint_embeddings_graph,
    criterion_cls: nn.Module,
    criterion_bound: nn.Module,
    lambda_bound_loss: float,
    device: str,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    boundary_th: float,
    tolerance: int,
    refinement_method: Optional[str] = None
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    losses = AverageMeter("Loss", ":.4e")
    postprocessor = PostProcessor(refinement_method, boundary_th)
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            # num = x.shape[0]
            batch_size = x.shape[1]

            output_cls, output_bound = model.forward(x, mask, joint_embeddings_graph)

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            t= t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )
            # update score
            scores_cls.update(output_cls, t, output_bound, mask)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

    cls_acc, edit_score, segment_f1s = scores_after_refinement.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s,
        bound_acc,
        precision,
        recall,
        bound_f1s,
    )


def get_mean_feature(feature, label, n_classes):

    feature_class_list = np.array([]).reshape(-1, 64)
    label_list = []
    for i in range(n_classes):
        class_indices = np.where(label == i)[0]
        if len(class_indices) > 0:
            feature_class = feature[class_indices]
            feature_class = np.mean(feature_class, axis=0, keepdims=True)
            feature_class_list = np.vstack([feature_class_list, feature_class])
            label_list.append(i)

    label_list = np.array(label_list).reshape(-1,1)
    return feature_class_list, label_list

def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
     joint_embeddings_graph,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    config : str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            num = x.shape[0]

            # compute output and loss
            output_cls, output_bound = model(x, mask, joint_embeddings_graph)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b= b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)

    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )
    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))
    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )