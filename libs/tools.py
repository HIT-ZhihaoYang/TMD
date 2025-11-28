import numpy as np
import torch

def segment_video_labels(video_labels_batch):
    segments_batch = []

    for video_labels in video_labels_batch:
        segments = []
        current_label = None
        segment_start = 0
        for i, label in enumerate(video_labels):
            label = label.item()
            if label != current_label:
                if (current_label is not None) and (label !=255):
                    segment_length = i - segment_start
                    segments.append((current_label, segment_length, segment_start, i - 1))
                current_label = label
                segment_start = i

        # 处理最后一个段
        if (current_label is not None) and (video_labels[-1] !=255):
            segment_length = len(video_labels) - segment_start
            segments.append((current_label, segment_length, segment_start, len(video_labels) - 1))
        segments_batch.append(segments)

    return segments_batch


def split_mixed_class(segments_batch, count):
    labels = []
    start_ends_batch = []
    for segments in segments_batch:
        # labels = []
        start_ends = []
        for i in range(len(segments)-count+1):
            label = [segments[j][0] for j in range(i, i+count)]
            start_end = [segments[i][2], segments[i+count-1][3]]
            labels.append(label)
            start_ends.append(start_end)

        start_ends_batch.append(start_ends)

    return labels, start_ends_batch


def split_feature(feature, device, num_segments=16):

    #将输入的 torch tensor 分割成指定数量的段

    N, C, T = feature.size()

    # 计算每个段的长度
    segment_length = T // num_segments

    features_split = []

    for i in range(N):
        for j in range(num_segments-1):
            # 提取每一段的帧级特征，并计算平均值
            split_feature = feature[i, :, j*segment_length:(j+1)*segment_length].mean(dim=-1)
            features_split.append(split_feature)
        features_split.append(feature[i, :, (num_segments-1)*segment_length:-1].mean(dim=-1))

    # 将段级特征列表转换为张量
    features_split = torch.stack(features_split).to(device)

    return features_split

def split_gt(gt, device, num_segments=16):

    #将输入的 torch tensor 分割成指定数量的段

    N, T = gt.size()

    # 计算每个段的长度
    segment_length = T // num_segments

    gts_split = []

    for i in range(N):
        for j in range(num_segments-1):
            # 提取每一段的帧级特征，并计算平均值
            split_gt = gt[i, j*segment_length:(j+1)*segment_length]
            gt_split = []
            last_element = split_gt[0]
            gt_label = [last_element]
            for element in split_gt:
                if (element != last_element) and (element != 255):
                    gt_label.append(element)
                    last_element = element

            gt_split = torch.stack(gt_label).to(device)
            if (gt_split == 255).any().item() == False:
                gts_split.append(gt_split)


        split_gt = gt[i, (num_segments-1)*segment_length:-1]
        gt_split = []
        last_element = split_gt[0]
        gt_label = [last_element]
        for element in split_gt:
            if (element != last_element) and (element != 255):
                gt_label.append(element)
                last_element = element

        gt_split = torch.stack(gt_label).to(device)
        if (gt_split == 255).any().item() == False:
            gts_split.append(gt_split)

    return gts_split


def split_gt_feature(gt, feature, device, num_segments=16):
    # 将输入的 torch tensor 分割成指定数量的段

    N, T = gt.size()

    # 计算每个段的长度
    segment_length = T // num_segments

    gts_split = []
    features_split = []

    for i in range(N):
        for j in range(num_segments - 1):
            # 提取每一段的帧级特征，并计算平均值
            split_gt = gt[i, j * segment_length:(j + 1) * segment_length]
            gt_split = []
            last_element = split_gt[0]
            gt_label = [last_element]
            for element in split_gt:
                if (element != last_element) and (element != 255):
                    gt_label.append(element)
                    last_element = element

            gt_split = torch.stack(gt_label).to(device)
            if (gt_split == 255).any().item() == False:
                gts_split.append(gt_split)
                split_feature = feature[i, :, j * segment_length:(j + 1) * segment_length].mean(dim=-1)
                features_split.append(split_feature)

        split_gt = gt[i, (num_segments - 1) * segment_length:-1]
        gt_split = []
        last_element = split_gt[0]
        gt_label = [last_element]
        for element in split_gt:
            if (element != last_element) and (element != 255):
                gt_label.append(element)
                last_element = element

        gt_split = torch.stack(gt_label).to(device)
        if (gt_split == 255).any().item() == False:
            gts_split.append(gt_split)
            features_split.append(feature[i, :, (num_segments - 1) * segment_length:-1].mean(dim=-1))

    # 将段级特征列表转换为张量
    features_split = torch.stack(features_split).to(device)

    return gts_split, features_split

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num)) #（N，N）
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label: #把同样都是这个类的都置为1
                gt[i,k] = 1
            # if a tuple
            # if isinstance(label, tuple) and labels[k] == label[::-1]:
            #     gt[i, k] = -1

    return gt #对比矩阵的gt


def gen_label_split(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num)) #（N，N）
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label: #把同样都是这个类的都置为1
                gt[i,k] = 1
    return gt #对比矩阵的gt

def generate_segment_features(output_feature, t_segment, device):
    segment_features = []

    for i in range(len(t_segment)):
        segment_list = t_segment[i]

        for j in range(len(segment_list)):
            start_frame = segment_list[j][2]
            end_frame = segment_list[j][3] + 1

            # 提取每一段的帧级特征，并计算平均值
            segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=-1)

            # 添加到段级特征列表中
            segment_features.append(segment_feature)

    # 将段级特征列表转换为张量
    segment_features = torch.stack(segment_features).to(device)

    return segment_features

def generate_transition_features(output_feature, t_segment, device):
    segment_features = []
    maintain_features = []

    for i in range(len(t_segment)):
        segment_list = t_segment[i]
        L = len(segment_list)
        if L > 1:
            for j in range(L-1):
                start_frame = segment_list[j][3] - segment_list[j][1] // 3
                end_frame = segment_list[j+1][2] +  segment_list[j+1][1] // 3 + 1

                # 提取每一段的帧级特征，并计算平均值
                segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=-1)

                # 添加到段级特征列表中
                segment_features.append(segment_feature)

                if j ==0:
                    maintain_feature = output_feature[i, :, :start_frame].mean(dim=-1)
                    maintain_features.append(maintain_feature)
                else:
                    m_start_frame = segment_list[j][2] + segment_list[j][1] // 3
                    m_end_frame = segment_list[j][3] - segment_list[j][1] // 3 + 1
                    maintain_feature = output_feature[i, :, m_start_frame:m_end_frame].mean(dim=-1)
                    maintain_features.append(maintain_feature)

            #last
            m_start_frame = segment_list[L-1][2] + segment_list[L-1][1] // 3
            maintain_feature = output_feature[i, :, m_start_frame:].mean(dim=-1)
            maintain_features.append(maintain_feature)
        elif L == 1:
            maintain_feature = output_feature[i, :, :].mean(dim=-1)
            maintain_features.append(maintain_feature)

    # 将段级特征列表转换为张量
    segment_features = torch.stack(segment_features).to(device)
    maintain_features = torch.stack(maintain_features).to(device)

    return segment_features, maintain_features

def generate_split_features(output_feature, start_ends, device):
    segment_features = []

    for i in range(len(start_ends)):
        segment_list = start_ends[i]

        for j in range(len(segment_list)):
            start_frame = segment_list[j][0]
            end_frame = segment_list[j][1] + 1

            # 提取每一段的帧级特征，并计算平均值
            segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=-1)

            # 添加到段级特征列表中
            segment_features.append(segment_feature)

    # 将段级特征列表转换为张量
    segment_features = torch.stack(segment_features).to(device)

    return segment_features

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True) #归一化为模长为1 在512维度上
    x2 = x2 / x2.norm(dim=-1, keepdim=True) #归一化为模长为1

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t() #余弦相似度 模为1,所以直接矩阵相乘就可以
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2 #两个分别是x1对x2的预先相似度与反之

def gen_transition_label(segments_batch):
    #every batch
    transition_label = list()
    for seg in segments_batch:
        action_label = [i[0] for i in seg]
        if len(action_label) > 1:
            transition_label.extend(
               [(action_label[j], action_label[j+1]) for j in range(len(action_label)-1)]
            )

    return transition_label


