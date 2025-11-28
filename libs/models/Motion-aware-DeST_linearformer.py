from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
# from .tcn import DilatedResidualLayer
from .SP import MultiScale_GraphConv
from .graph.graph import Graph
from .graph.tools import k_adjacency, normalize_adjacency_matrix, get_adjacency_matrix
from libs.models.motion_encoder import StateEncoder, TemporalEncoder, SpatialEncoder ,CJM,CrossXMFusion
from libs.helper import split_and_pad_tensor
def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

def positional_encoding(seq_len, d_model):
    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel)
        self.dropout = nn.Dropout(drop_out)

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1
        
    def forward(self, queries, keys, values, mask):

        B, L, D = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
        values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values)  
        Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6)

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) 
 
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)

        return x * mask[:, 0, :, None]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            )
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x)
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
       
        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1) 
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features))
        
    def forward(self, feature_s, feature_t, mask):
        feature_s = feature_s.permute(0, 2, 1)
        n, c, t = feature_s.shape
        feature_s = self.conv_s(feature_s)
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t)/t)
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t)
        feature_cross = feature_cross + feature_t
        feature_cross = feature_cross.permute(0, 2, 1)
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask
    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.4, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer
        num_SFI_layers = len(SFI_layer)
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)

        self.conv_in = nn.Conv2d(in_channel, num_SFI_layers+1, kernel_size=1)
        self.conv_t = nn.Conv1d(node, n_features, 1)
        self.SFI_layers = nn.ModuleList(
            [SFI(node, n_features) for i in range(num_SFI_layers)])
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in
                range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        count = 0
        x = self.conv_in(x)
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers), 1), dim=1)
        feature_t = feature_t.squeeze(1).permute(0, 2, 1)
        feature_st = self.conv_t(feature_t)

        for index, layer in enumerate(self.layers):
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count,:], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, None, mask)

        feature_st = self.conv_out(feature_st)
        return feature_st * mask

class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in 
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        
        return out, feature

class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature)
        return out * mask[:, 0:1, :], feature

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)

        return (x + out) * mask[:, 0:1, :]

class MotionModule(nn.Module):
    def __init__(self, in_channel, n_features, historical_steps, ZL, dataset, disentangled_agg=True, num_scales=13,
                 use_mask=True,):
        super().__init__()

        self.steps = historical_steps
        self.fc = nn.Linear(in_channel, n_features // 2, bias=False) #6 * 6
        self.fc2 = nn.Linear(n_features // 2, n_features, bias=False)
        self.sencoder = StateEncoder(in_channel, n_features)
        self.tpe = nn.Embedding(10000, n_features)
        self.temporal = TemporalEncoder(historical_steps=historical_steps,
                                        embed_dim=n_features,
                                        dropout=0.1,
                                        num_layers=2)
        self.spatial = SpatialEncoder(n_features)
        self.ZL = ZL


    def forward(self, motion, x):
        B, D, V, T = motion.shape
        motion = motion.permute(0, 2, 3, 1).contiguous().view(-1, T, D)  # BVTD
        motion = self.fc2(F.relu(self.fc(motion)))
        motion = motion.view(B, T, V, -1)  # BTVD

        fr_diff = torch.arange(T).unsqueeze(0).expand(B, T).to(motion.device)
        fr_diff = self.tpe(fr_diff).unsqueeze(2).expand(-1, -1, V, -1)
        motion = motion + fr_diff
        motion, abdt = split_and_pad_tensor(motion.permute(0, 2, 3, 1), B=self.steps)  # NBVDT
        x, _ = split_and_pad_tensor(x, B=self.steps)

        N, B, V, D, T = motion.shape
        motion= self.temporal(motion.permute(4, 0, 1, 2,3).contiguous().view(self.steps, N * B * V, -1),
                                   torch.zeros(N * B * V, self.steps).bool().cuda())
        # motion = motion[:,:,:,:,T//2].reshape(-1,D)
        pre_last = x[:, :, :, :, self.steps-1].permute(0, 1,3,2).contiguous().view(N * B * V, -1)
        pre_last = self.sencoder(pre_last)
        motion = motion.view(N*B, V, -1)
        pre_last = pre_last.view(N*B, V, -1)
        motion = self.spatial(motion, motion, pre_last, None, None).squeeze(1)
        motion = motion.unsqueeze(1).repeat(1, self.steps, 1, 1)

        motion = motion.view(N, B, T, V, D).permute(1,4,0,2,3).contiguous().view(B,D,N*T,V)
        if abdt != 0:
            motion = motion[:, :, :-abdt, :]

        return motion


class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        motion_channel: int,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        step:int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        SFI_layer: Optional[int] = None,
        dataset: str = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()

        self.motion_channel = motion_channel
        self.in_channel = in_channel
        if dataset == "LARA" :
            self.node = 19
        elif dataset == "TCG-15":
            self.node = 17
        else:
            self.node = 25

        self.logit_scale = nn.Parameter(torch.ones(2) * np.log(1 / 0.07))  # 2.6593
        self.fuse_x = nn.Parameter(torch.tensor(0.0))
        self.fuse_m = nn.Parameter(torch.tensor(0.0))
        self.dataset = dataset
        self.n_features = n_features
        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset)  
        self.STI = STI(self.node, n_features, n_features, n_features, n_layers, SFI_layer)
        self.motion = MotionModule(motion_channel, n_features, step, 0, dataset)
        self.cjm = CJM(self.n_features, self.n_features,self.n_features)
        self.crossfusion = CrossXMFusion(self.n_features)
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)
        self.conv_feature = nn.Conv1d(n_features, 768, 1)  # 
        self.conv_motion = nn.Conv1d(n_features, 768, 1)  # 

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        conv_asb_feature = [nn.Conv1d(n_features, 768, 1) for s in range(n_stages_asb - 1)]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]

        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)
        self.conv_asb_feature = nn.ModuleList(conv_asb_feature)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()


    def forward(self, input: torch.Tensor, mask: torch.Tensor, joint_graph) -> Tuple[torch.Tensor, torch.Tensor]:
        motion, x = input[:,0:self.in_channel:,:], input[:,self.in_channel:,:,:]
        motion_s =  self.motion(motion, x) * mask.unsqueeze(3)
        x = self.SP(x, joint_graph) * mask.unsqueeze(3)
        feature_x, feature_m = self.cjm(x,motion_s)

        B, D, T, V = feature_x.shape

        x_motion = torch.cat((feature_x,feature_m), dim=0)
        mask_2b = torch.cat((mask,mask),dim=0)
        x_motion = self.STI(x_motion, mask_2b)
        feature_x, feature_m = x_motion[0:B,:,:], x_motion[B:,:,:]

        feature_x, feature_m = self.crossfusion(feature_x, feature_m)
        out_cls = self.conv_cls(feature_x)
        out_bound = self.conv_bound(feature_m)
        out_feature = self.conv_feature(feature_x)  
        out_motion = self.conv_motion(feature_m)

        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]
            outputs_feature = [out_feature]
            outputs_motion = [out_motion]

            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature_x = as_stage(self.activation_asb(out_cls)* mask, feature_x* mask, mask)
                out_feature = conv_stage(feature_x)
                outputs_cls.append(out_cls)
                outputs_feature.append(out_feature)

            for br_stage in self.brb:
                out_bound, feature_m = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound, outputs_feature,outputs_motion, self.logit_scale)
        else:
            for as_stage in self.asb:
                out_cls, _ = as_stage(self.activation_asb(out_cls)* mask, feature_x* mask, mask)

            for br_stage in self.brb:
                out_bound, _ = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound)
