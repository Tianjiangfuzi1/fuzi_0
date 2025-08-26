# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------


from random import sample
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import pdb
import math
import pickle
import time
import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmcv.parallel import DataContainer as DC
from mmseg.datasets.pipelines.formatting import to_tensor
from ..builder import build_loss, build_backbone, build_neck
from .transformer import TransformerEncoderLayer, TransformerEncoder
from .position_encoding import PositionEmbeddingSine
from ..builder import LANENET2S
from .tools import homography_crop_resize
from .utils import AnchorGenerator, nms_3d
from thop import profile
from mmcv import Config
from model.head.detector_head import bulid_head
from structures.image_list import to_image_list
from engine.inference import inference, inference_all_depths
from utils import comm
from utils.miscellaneous import mkdir
from data.datasets.evaluation import evaluate_python
from data.datasets.evaluation import generate_kitti_3d_detection
import os
import linecache
from itertools import islice
from thop import profile
from structures.params_3d import ParamsList
from data.datasets.kitti_utils import Calibration, read_label, approx_proj_center, refresh_attributes, show_heatmap, show_image_with_boxes, show_edge_heatmap
from mmcv.ops.point_sample import bilinear_grid_sample
from PIL import Image
from model.heatmap_coder import (
gaussian_radius,
draw_umich_gaussian,
draw_gaussian_1D,
draw_ellip_gaussian,
draw_umich_gaussian_2D,
)
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
import gc
class DecodeLayer(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(DecodeLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, out_channel))
    def forward(self, x):
        return self.layer(x)

@LANENET2S.register_module()
class Anchor3DLane(BaseModule):

    def __init__(self,
                 backbone,
                 neck = None,
                 pretrained = None,
                 y_steps = [  5.,  10.,  15.,  20.,  30.,  40.,  50.,  60.,  80.,  100.],
                 feat_y_steps = [  5.,  10.,  15.,  20.,  30.,  40.,  50.,  60.,  80.,  100.],
                 anchor_cfg = None,
                 db_cfg = None,
                 backbone_dim = 512,
                 attn_dim = None,
                 iter_reg = 0,
                 # drop_out = 0.1,
                 # num_heads = None,
                 # enc_layers = 1,
                 # dim_feedforward = None,
                 # pre_norm = None,
                 anchor_feat_channels = 64,
                 feat_size = (48, 60),
                 num_category = 21,
                 loss_lane = None,
                 loss_aux = None,
                 init_cfg = None,
                 train_cfg = None,
                 test_cfg = None,
                 idx = None):
        super(Anchor3DLane, self).__init__(init_cfg)
        assert loss_aux is None or len(loss_aux) == iter_reg
        # monoflex cfg
        # Anchor3DLane setting
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.db_cfg = db_cfg
        hidden_dim = attn_dim
        self.iter_reg = iter_reg
        self.loss_aux = loss_aux
        self.anchor_feat_channels = anchor_feat_channels
        self.feat_size = feat_size
        self.num_category = num_category
        # self.enc_layers = enc_layers
        self.fp16_enabled = False

        # Anchor
        self.y_steps = np.array(y_steps, dtype=np.float32)
        self.feat_y_steps = np.array(feat_y_steps, dtype=np.float32)
        self.feat_sample_index = torch.from_numpy(np.isin(self.y_steps, self.feat_y_steps))
        self.x_norm = 30.
        self.y_norm = 100.
        self.z_norm = 10.
        self.x_min = -30
        self.x_max = 30
        self.anchor_len = len(y_steps)
        self.anchor_feat_len = len(feat_y_steps)
        
        #车道线锚点：
        self.anchor_generator = AnchorGenerator(anchor_cfg, x_min=self.x_min, x_max=self.x_max, y_max=int(self.y_steps[-1]),
                                                norm=(self.x_norm, self.y_norm, self.z_norm)) 
        dense_anchors = self.anchor_generator.generate_anchors()  # [N, 65]
        anchor_inds = self.anchor_generator.y_steps   # [100] 
        self.anchors = self.sample_from_dense_anchors(self.y_steps, anchor_inds, dense_anchors)
        self.feat_anchors = self.sample_from_dense_anchors(self.feat_y_steps, anchor_inds, dense_anchors)
        self.xs, self.ys, self.zs = self.compute_anchor_cut_indices(self.feat_anchors, self.feat_y_steps)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        # transformer layer
        # self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2, normalize=True)
        # self.input_proj = nn.Conv2d(backbone_dim, hidden_dim, kernel_size=1)  # the same as channel of self.layer4
        # if self.enc_layers == 1:
        #     self.transformer_layer = TransformerEncoderLayer(hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
        #                             dropout=drop_out, normalize_before=pre_norm)
        # else:
        #     transformer_layer = TransformerEncoderLayer(hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, \
        #         dropout=drop_out, normalize_before=pre_norm)
        #     self.transformer_layer = TransformerEncoder(transformer_layer, self.enc_layers)
                                        
        # decoder heads
        self.anchor_projection = nn.Conv2d(hidden_dim, self.anchor_feat_channels, kernel_size=1)

        # FPN
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None

        self.cls_layer = nn.ModuleList()
        self.reg_x_layer = nn.ModuleList()
        self.reg_z_layer = nn.ModuleList()
        self.reg_vis_layer = nn.ModuleList()

        self.cls_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels * self.anchor_feat_len, self.num_category))
        self.reg_x_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
        self.reg_z_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
        self.reg_vis_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))

        # build loss function
        self.lane_loss = build_loss(loss_lane)

        # build iterative regression layers
        self.build_iterreg_layers()
        DATASETS = dict(
            DETECT_CLASSES=("Car", "Pedestrian", "Cyclist"),
            TRAIN=("kitti_train",),
            TEST=("kitti_train",),
            TRAIN_SPLIT="train",
            TEST_SPLIT="val",
            USE_RIGHT_IMAGE=False,
            CONSIDER_OUTSIDE_OBJS=True,
            FILTER_ANNO_ENABLE=True,
            FILTER_ANNOS=[0.9, 20],
            MAX_OBJECTS=40,
            MIN_RADIUS=0.0,
            MAX_RADIUS=0.0,
            CENTER_RADIUS_RATIO=0.1
        )

        DATALOADER = dict(
            NUM_WORKERS=4,
            SIZE_DIVISIBILITY=0,
            ASPECT_RATIO_GROUPING=False,

        )

        INPUT = dict(
            TO_BGR=False,
            PIXEL_MEAN=[0.485, 0.456, 0.406],
            PIXEL_STD=[0.229, 0.224, 0.225],
            # Size of the smallest side of the image during training
            HEIGHT_TRAIN=320,
            # Maximum size of the side of the image during training
            WIDTH_TRAIN=480,
            # Size of the smallest side of the image during testing
            HEIGHT_TEST=320,
            # Maximum size of the side of the image during testing
            WIDTH_TEST=480,

            HEATMAP_CENTER='3D',
            AUG_PARAMS=[[0.5]],

            ORIENTATION='multi-bin',
            MODIFY_ALPHA=False,

            ORIENTATION_BIN_SIZE=4,
            APPROX_3D_CENTER='intersect',

            ADJUST_BOUNDARY_HEATMAP=True,
            KEYPOINT_VISIBLE_MODIFY=True,
            USE_APPROX_CENTER=False,
            ADJUST_DIM_HEATMAP=False,
            HEATMAP_RATIO=0.5,
            ELLIP_GAUSSIAN=False,
            IGNORE_DONT_CARE=False,
        )

        MODEL = dict(
            INPLACE_ABN=True,
            DEVICE="cuda",
            BACKBONE=dict(
                CONV_BODY="dla34",
                # Add StopGrad at a specified stage so the bottom layers are frozen
                FREEZE_CONV_BODY_AT=0,
                # Normalization for backbone
                DOWN_RATIO=4
            ),
            HEAD=dict(
                REGRESSION_HEADS=[['2d_dim'], ['3d_offset'], ['corner_offset'], ['corner_uncertainty'], ['3d_dim'],
                                  ['ori_cls', 'ori_offset'], ['depth'], ['depth_uncertainty']],
                REGRESSION_CHANNELS=[[4, ], [2, ], [20], [3], [3, ], [8, 8], [1, ], [1, ]],

                ENABLE_EDGE_FUSION=True,
                TRUNCATION_OUTPUT_FUSION='add',
                EDGE_FUSION_NORM='BN',
                TRUNCATION_OFFSET_LOSS='log',

                BN_MOMENTUM=0.1,
                USE_NORMALIZATION="BN",
                LOSS_TYPE=["Penalty_Reduced_FocalLoss", "L1", "giou", "L1"],

                MODIFY_INVALID_KEYPOINT_DEPTH=True,
                CORNER_LOSS_DEPTH='soft_combine',
                LOSS_NAMES=['hm_loss', 'bbox_loss', 'depth_loss', 'offset_loss', 'orien_loss', 'dims_loss',
                            'corner_loss',
                            'keypoint_loss', 'keypoint_depth_loss', 'trunc_offset_loss', 'weighted_avg_depth_loss'],
                LOSS_UNCERTAINTY=[True, True, False, True, True, True, True, True, False, True, True],
                INIT_LOSS_WEIGHT=[1, 1, 1, 0.5, 1, 1, 0.2, 1.0, 0.2, 0.1, 0.2],

                CENTER_MODE='max',
                HEATMAP_TYPE='centernet',
                DIMENSION_REG=['exp', True, False],
                USE_UNCERTAINTY=False,
                DEPTH_MODE='inv_sigmoid',
                OUTPUT_DEPTH='soft',
                DIMENSION_WEIGHT=[1, 1, 1],
                UNCERTAINTY_INIT=True,
                REDUCE_LOSS_NORM=True,
                USE_SYNC_BN=True,
                PREDICTOR = "Base_Predictor",
                NUM_CHANNEL = 256,
                INIT_P=0.01,
                EDGE_FUSION_RELU=False,
                EDGE_FUSION_KERNEL_SIZE=3,
                DEPTH_RANGE = [0.1, 100],
                DEPTH_REFERENCE = (26.494627, 16.05988),
                SUPERVISE_CORNER_DEPTH = False,
                REGRESSION_OFFSET_STAT = [-0.5844396972302358, 9.075032501413093],
                REGRESSION_OFFSET_STAT_NORMAL = [-0.01571878324572745, 0.05915441457040611],
                CENTER_SAMPLE='center',
                CENTER_AGGREGATION = False,
                # for vanilla focal loss
                LOSS_ALPHA = 0.25,
                LOSS_GAMMA = 2,
                # for penalty-reduced focal loss
                LOSS_PENALTY_ALPHA = 2,
                LOSS_BETA = 4,
                # 2d offset, 2d dimension
                BIAS_BEFORE_BN = False,
                UNCERTAINTY_RANGE = [-10, 10],
                UNCERTAINTY_WEIGHT = 1.0,
                KEYPOINT_LOSS = 'L1',
                KEYPOINT_NORM_FACTOR = 1.0,
                KEYPOINT_XY_WEIGHT = [1, 1],
                DEPTH_FROM_KEYPOINT = False,
                KEYPOINT_TO_DEPTH_RELU = True,
                REGRESSION_AREA = False,
                # edge fusion module
                TRUNCATION_CLS = False,
                DIMENSION_MEAN=((3.8840, 1.5261, 1.6286),
                                (0.8423, 1.7607, 0.6602),
                                (1.7635, 1.7372, 0.5968)),

                DIMENSION_STD=((0.4259, 0.1367, 0.1022),
                               (0.2349, 0.1133, 0.1427),
                               (0.1766, 0.0948, 0.1242)),

            )

        )
        TEST = dict(
            IMS_PER_BATCH=1,
            UNCERTAINTY_AS_CONFIDENCE=True,
            DETECTIONS_THRESHOLD=0.2,
            METRIC=['R40'],
            SINGLE_GPU_TEST = True,
            PRED_2D = True,
            EVAL_DIS_IOUS = False,
            EVAL_DEPTH = False,
            EVAL_DEPTH_METHODS = [],
        # 'none', '2d', '3d'
            USE_NMS = 'none',
            NMS_THRESH = -1.,
            NMS_CLASS_AGNOSTIC = False,

        # Number of detections per image
            DETECTIONS_PER_IMG = 50,
            VISUALIZE_THRESHOLD = 0.4,
        )
        cfg = {
            'INPUT': INPUT,
            'MODEL': MODEL,
            'DATASETS': DATASETS,
            'TEST': TEST
        }
        cfg = Config(cfg)
        self.heads = bulid_head(cfg, 64)
        self.idx = idx
        self.reduce_conv = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)

    def build_iterreg_layers(self):
        self.aux_loss = nn.ModuleList()
        for iter in range(self.iter_reg):
            self.cls_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels * self.anchor_feat_len, self.num_category))
            self.reg_x_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.reg_z_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.reg_vis_layer.append(DecodeLayer(self.anchor_feat_channels * self.anchor_feat_len, self.anchor_feat_channels, self.anchor_len))
            self.aux_loss.append(build_loss(self.loss_aux[iter]))

    def sample_from_dense_anchors(self, sample_steps, dense_inds, dense_anchors):
        sample_index = np.isin(dense_inds, sample_steps)
        anchor_len = len(sample_steps)
        dense_anchor_len = len(sample_index)
        anchors = np.zeros((len(dense_anchors), 5 + anchor_len * 3), dtype=np.float32)
        anchors[:, :5] = dense_anchors[:, :5].copy()
        anchors[:, 5:5 + anchor_len] = dense_anchors[:, 5:5 + dense_anchor_len][:, sample_index]  # [N, 20]
        anchors[:, 5 + anchor_len:5 + 2 * anchor_len] = dense_anchors[:, 5 + dense_anchor_len:5 + 2 * dense_anchor_len][
                                                        :, sample_index]  # [N, 20]
        anchors = torch.from_numpy(anchors)
        return anchors

    def compute_anchor_cut_indices(self, anchors, y_steps):
        # definitions
        if len(anchors.shape) == 2:
            n_proposals = len(anchors)
        else:
            batch_size, n_proposals = anchors.shape[:2]

        num_y_steps = len(y_steps)

        # indexing
        xs = anchors[..., 5:5 + num_y_steps]  # [N, l] or [B, N, l]
        xs = torch.flatten(xs, -2)  # [Nl] or [B, Nl]

        ys = torch.from_numpy(y_steps).to(anchors.device)   # [l]
        if len(anchors.shape) == 2:
            ys = ys.repeat(n_proposals)  # [Nl]
        else:
            ys = ys.repeat(batch_size, n_proposals)  # [B, Nl]

        zs = anchors[..., 5 + num_y_steps:5 + num_y_steps * 2]  # [N, l]
        zs = torch.flatten(zs, -2)  # [Nl] or [B, Nl]
        return xs, ys, zs


    def projection_transform(self, Matrix, xs, ys, zs):
        # Matrix: [B, 3, 4], x, y, z: [B, NCl]
        ones = torch.ones_like(zs)   # [B, NCl]
        # ones = torch.full((1, 44310), 1)
        coordinates = torch.stack([xs, ys, zs, ones], dim=1)   # [B, 4, NCl]
        trans = torch.bmm(Matrix, coordinates)   # [B, 3, NCl]

        u_vals = trans[:, 0, :] / trans[:, 2, :]   # [B, NCl]
        v_vals = trans[:, 1, :] / trans[:, 2, :]   # [B, NCl]
        return u_vals, v_vals

    def cut_anchor_features(self, features, h_g2feats, xs, ys, zs):
        # definitions
        batch_size = features.shape[0]

        if len(xs.shape) == 1:
            batch_xs = xs.repeat(batch_size, 1)  # [B, Nl]
            batch_ys = ys.repeat(batch_size, 1)  # [B, Nl]
            batch_zs = zs.repeat(batch_size, 1)  # [B, Nl]
        else:
            batch_xs = xs
            batch_ys = ys
            batch_zs = zs

        batch_us, batch_vs = self.projection_transform(h_g2feats, batch_xs, batch_ys, batch_zs)
        batch_us = (batch_us / self.feat_size[1] - 0.5) * 2
        batch_vs = (batch_vs / self.feat_size[0] - 0.5) * 2

        batch_grid = torch.stack([batch_us, batch_vs], dim=-1)  #
        batch_grid = batch_grid.reshape(batch_size, -1, self.anchor_feat_len, 2)  # [B, N, l, 2]
        batch_anchor_features = F.grid_sample(features, batch_grid, padding_mode='zeros')  # [B, C, N, l]
        # batch_anchor_features = self.bilinear_grid_sample(features, batch_grid.float())
        valid_mask = (batch_us > -1) & (batch_us < 1) & (batch_vs > -1) & (batch_vs < 1)

        return batch_anchor_features, valid_mask.reshape(batch_size, -1, self.anchor_feat_len)

    #双线性插值到 (80,120)（你的配置里就是这个 feat_size）：feat = output[-1]（或 neck 的 output[0]）→ trans_feat = self.reduce_conv(feat) → F.interpolate(..., size=(80,120))
    def feature_extractor(self, img, mask):
        output = self.backbone(img)
        if self.neck is not None:
            output = self.neck(output)
            feat = output[0]
        else:
            feat = output[-1]
        trans_feat = self.reduce_conv(feat)
        #看看原始大小以及为什么要双线性插值到80,120
        trans_feat = F.interpolate(trans_feat, size=(80, 120), mode='bilinear', align_corners=False)
        # trans_feat = output
        return trans_feat

    @force_fp32()
    def get_proposals(self, project_matrixes, anchor_feat, iter_idx=0, targets=None, proposals_prev=None):
        #解码器由 4 组 MLP 组成：类别 cls_layer、x 偏移 reg_x_layer、z 偏移 reg_z_layer、可见性 reg_vis_layer，每组都是 DecodeLayer(线性+ReLU6×2+线性) 并按 iter_idx 支持迭代回归（iter_reg 次） 。
        #get_proposals()：把取到的每条锚线的特征拼成 [B*N, C*l]，分别过 4 个 MLP，reg_vis 过 sigmoid，然后加到锚点得到最终 proposal：
        #cls_logits/reg_x/reg_z/reg_vis = ... → 写入 reg_proposals（形状包含 5 + 3*anchor_len + num_category） 。
        #训练时用 self.lane_loss = build_loss(loss_lane) 计算车道线损失；若启用多次迭代，会额外加上 aux_loss（build_iterreg_layers()）
        
        batch_size = project_matrixes.shape[0]
        if proposals_prev is None:
            batch_anchor_features, _ = self.cut_anchor_features(anchor_feat, project_matrixes, self.xs, self.ys, self.zs)  # [B, C, N, l]
        else:
            sampled_anchor = torch.zeros(batch_size, len(self.anchors), 5 + self.anchor_feat_len * 3,
                                         device=anchor_feat.device)
            sampled_anchor[:, :, 5:5 + self.anchor_feat_len] = proposals_prev[:, :, 5:5 + self.anchor_len][:, :,
                                                               self.feat_sample_index]
            sampled_anchor[:, :, 5 + self.anchor_feat_len:5 + self.anchor_feat_len * 2] = proposals_prev[:, :,
                                                                                          5 + self.anchor_len:5 + self.anchor_len * 2][
                                                                                          :, :, self.feat_sample_index]
            xs, ys, zs = self.compute_anchor_cut_indices(sampled_anchor, self.feat_y_steps)
            batch_anchor_features, _ = self.cut_anchor_features(anchor_feat, project_matrixes, xs, ys,
                                                                zs)  # [B, C, N, l]

        batch_anchor_features = batch_anchor_features.transpose(1, 2)  # [B, N, C, l]
        batch_anchor_features = batch_anchor_features.reshape(-1, self.anchor_feat_channels * self.anchor_feat_len)  # [B * N, C * l]

        # Predict
        cls_logits = self.cls_layer[iter_idx](batch_anchor_features)  # [B * N, C]
        cls_logits = cls_logits.reshape(batch_size, -1, cls_logits.shape[1])  # [B, N, C]
        reg_x = self.reg_x_layer[iter_idx](batch_anchor_features)  # [B * N, l]
        reg_x = reg_x.reshape(batch_size, -1, reg_x.shape[1])  # [B, N, l]
        reg_z = self.reg_z_layer[iter_idx](batch_anchor_features)  # [B * N, l]
        reg_z = reg_z.reshape(batch_size, -1, reg_z.shape[1])  # [B, N, l]
        reg_vis = self.reg_vis_layer[iter_idx](batch_anchor_features)  # [B * N, l]
        reg_vis = torch.sigmoid(reg_vis)
        reg_vis = reg_vis.reshape(batch_size, -1, reg_vis.shape[1])  # [B, N, l]

        # Add offsets to anchors
        # [B, N, l]
        reg_proposals = torch.zeros(batch_size, len(self.anchors), 5 + self.anchor_len * 3 + self.num_category,
                                    device=project_matrixes.device)
        if proposals_prev is None:
            reg_proposals[:, :, :5 + self.anchor_len * 3] = reg_proposals[:, :, :5 + self.anchor_len * 3] + self.anchors
        else:
            reg_proposals[:, :, :5 + self.anchor_len * 3] = reg_proposals[:, :, :5 + self.anchor_len * 3] + proposals_prev[:, :, :5 + self.anchor_len * 3]

        reg_proposals[:, :, 5:5 + self.anchor_len] += reg_x
        reg_proposals[:, :, 5 + self.anchor_len:5 + self.anchor_len * 2] += reg_z
        reg_proposals[:, :, 5 + self.anchor_len * 2:5 + self.anchor_len * 3] = reg_vis
        reg_proposals[:, :, 5 + self.anchor_len * 3:5 + self.anchor_len * 3 + self.num_category] = cls_logits  # [B, N, C]
        return reg_proposals

    def encoder_decoder(self, img, mask, gt_project_matrix, targets, **kwargs):
        # img: [B, 3, inp_h, inp_w]; mask: [B, 1, 36, 480]
        batch_size = img.shape[0]

        trans_feat = self.feature_extractor(img, mask)
        # anchor
        anchor_feat = self.anchor_projection(trans_feat)
        # 投影矩阵
        project_matrixes = self.obtain_projection_matrix(gt_project_matrix, self.feat_size)
        project_matrixes = torch.stack(project_matrixes, dim=0)   # [B, 3, 4]
        reg_proposals_all = []
        anchors_all = []
        # 得到 地面→特征 的矩阵 h_g2feat
        #再把每个锚点上的 (x,y,z) 投到特征图坐标 (u,v)，并构成 grid：projection_transform(...) → grid = [...].reshape(B, N, l, 2) → F.grid_sample(features, grid)，这样就获得“沿每条锚线各 y 采样点的特征序列”
        reg_proposals_s1 = self.get_proposals(project_matrixes, anchor_feat, 0, targets, None)
        reg_proposals_all.append(reg_proposals_s1)
        anchors_all.append(torch.stack([self.anchors] * batch_size, dim=0))
        for iter in range(self.iter_reg):
            proposals_prev = reg_proposals_all[iter]
            reg_proposals_all.append(self.get_proposals(project_matrixes, anchor_feat, iter+1, proposals_prev))
            anchors_all.append(proposals_prev[:, :, :5+self.anchor_len*3])

        output = {'reg_proposals': reg_proposals_all[-1], 'anchors':anchors_all[-1]}
        if self.training:
            loss_dict, log_loss_dict = self.heads(trans_feat, targets)
            if self.iter_reg > 0:
                output_aux = {'reg_proposals': reg_proposals_all[:-1], 'anchors': anchors_all[:-1]}
                return output, output_aux, loss_dict, log_loss_dict
            return output, None, loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.heads(trans_feat, targets, test=False)
            if self.iter_reg > 0:
                output_aux = {'reg_proposals': reg_proposals_all[:-1], 'anchors': anchors_all[:-1]}
                return output, output_aux, result, eval_utils, visualize_preds
            return output, None, result, eval_utils, visualize_preds

            # get onnx model
            # x = self.heads(trans_feat, targets, test=False)
            # if self.iter_reg > 0:
            #     output_aux = {'reg_proposals': reg_proposals_all[:-1], 'anchors': anchors_all[:-1]}
            #     return output, output_aux, x
            # return output, None, x

    def get_edge_utils(self, image_size, pad_size, down_ratio=4):
        img_w, img_h = image_size

        x_min, y_min = np.ceil(pad_size[0] / down_ratio), np.ceil(pad_size[1] / down_ratio)
        x_max, y_max = (pad_size[0] + img_w - 1) // down_ratio, (pad_size[1] + img_h - 1) // down_ratio

        step = 1
        # boundary idxs
        edge_indices = []

        # left
        y = torch.arange(y_min, y_max, step)
        x = torch.ones(len(y)) * x_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # bottom
        x = torch.arange(x_min, x_max, step)
        y = torch.ones(len(x)) * y_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0)
        edge_indices.append(edge_indices_edge)

        # right
        y = torch.arange(y_max, y_min, -step)
        x = torch.ones(len(y)) * x_max

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # top
        x = torch.arange(x_max, x_min - 1, -step)
        y = torch.ones(len(x)) * y_min

        edge_indices_edge = torch.stack((x, y), dim=1)
        edge_indices_edge[:, 0] = torch.clamp(edge_indices_edge[:, 0], x_min)
        edge_indices_edge[:, 1] = torch.clamp(edge_indices_edge[:, 1], y_min)
        edge_indices_edge = torch.unique(edge_indices_edge, dim=0).flip(dims=[0])
        edge_indices.append(edge_indices_edge)

        # concatenate
        edge_indices = torch.cat([index.long() for index in edge_indices], dim=0)

        return edge_indices

    def obtain_projection_matrix(self, project_matrix, feat_size):
        """
            Update transformation matrix based on ground-truth cam_height and cam_pitch
            This function is "Mutually Exclusive" to the updates of M_inv from network prediction
        :param args:
        :param cam_height:
        :param cam_pitch:
        :return:
        """
        #裁剪/缩放 单应性 Hc 相乘
        h_g2feats = []
        device = project_matrix.device
        project_matrix = project_matrix.cpu().numpy()
        for i in range(len(project_matrix)):
            P_g2im = project_matrix[i]
            # Openlane image
            Hc = homography_crop_resize((self.db_cfg.org_h, self.db_cfg.org_w), 0, feat_size)
            # self-collection image
            # Hc = homography_crop_resize((1080, 1620), 0, feat_size)
            h_g2feat = np.matmul(Hc, P_g2im)
            h_g2feats.append(torch.from_numpy(h_g2feat).type(torch.FloatTensor).to(device))
        return h_g2feats

    def nms(self, batch_proposals, batch_anchors, nms_thres=0, conf_threshold=None, refine_vis=False, vis_thresh=0.5):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, anchors in zip(batch_proposals, batch_anchors):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # The gradients do not have to (and can't) be calculated for the NMS procedure
            # apply nms
            scores = 1 - softmax(proposals[:, 5 + self.anchor_len * 3:5 + self.anchor_len * 3 + self.num_category])[:,0]  # pos_score  # for debug
            if conf_threshold > 0:
                above_threshold = scores > conf_threshold
                proposals = proposals[above_threshold]
                scores = scores[above_threshold]
                anchor_inds = anchor_inds[above_threshold]
            if proposals.shape[0] == 0:
                proposals_list.append((proposals[[]], anchors[[]], None))
                continue
            if nms_thres > 0:
                # refine vises to ensure consistent lane
                vises = proposals[:,
                        5 + self.anchor_len * 2:5 + self.anchor_len * 3] >= vis_thresh  # need check  #[N, l]
                flag_l = vises.cumsum(dim=1)
                flag_r = vises.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                refined_vises = (flag_l > 0) & (flag_r > 0)
                if refine_vis:
                    proposals[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3] = refined_vises
                keep = nms_3d(proposals, scores, refined_vises, thresh=nms_thres, anchor_len=self.anchor_len)
                proposals = proposals[keep]
                anchor_inds = anchor_inds[keep]
                proposals_list.append((proposals, anchors[anchor_inds], anchor_inds))
            else:
                proposals_list.append((proposals, anchors[anchor_inds], anchor_inds))
        return proposals_list


    def forward_dummy(self, img, mask=None, img_metas=None, gt_project_matrix=None,targets=None,**kwargs):
        mask = img.new_zeros((img.shape[0], 1, img.shape[2], img.shape[3]))
        gt_project_matrix = img.new_zeros((img.shape[0], 3, 4))
        output, _ = self.encoder_decoder(img, mask, gt_project_matrix,targets,**kwargs)
        return output

    def forward_test(self, img=None, mask=None, img_metas=None, images=None, targets=None, img_ids=None, gt_project_matrix=None, **kwargs):
        gt_project_matrix = gt_project_matrix.squeeze(1)
        output, _, result, eval_utils, visualize_preds= self.encoder_decoder(img, mask, gt_project_matrix, targets, **kwargs)
        proposals_list = self.nms(output['reg_proposals'], output['anchors'], self.test_cfg.nms_thres,
                                      self.test_cfg.conf_threshold, refine_vis=self.test_cfg.refine_vis,
                                      vis_thresh=self.test_cfg.vis_thresh)
        output['proposals_list'] = proposals_list
        cpu_device = torch.device("cpu")
        result = result.to(cpu_device)
        predict_txt = img_ids[0] + '.txt'
        parts = predict_txt.split('/')
        filename = parts[-2]
        txtname = parts[-1]
        predict_folder = '/root/autodl-tmp/Anchor3DLane/output/openlane/waymo-kitti/validation'
        path = os.path.join(predict_folder + '/' + filename)
        file_path = os.path.join(path + '/' + txtname)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(file_path, 'w') as f:
            f.write('')
        predict_folder1 = '/root/autodl-tmp/Anchor3DLane/output/openlane/waymo-kitti/'
        predict_txt = os.path.join(predict_folder1, predict_txt)
        generate_kitti_3d_detection(result, predict_txt)
        return output

        # get onnx model
        # gt_project_matrix = gt_project_matrix.squeeze(1)
        # output, _, x = self.encoder_decoder(img, mask, gt_project_matrix, targets, **kwargs)
        # return output, x

    def forward(self, img=None, img_metas=None, targets=None, mask=None, image_id=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        # self.idx = image_id
        if return_loss :
            device = torch.device("cuda")
            img = img.to(device)
            targets = [target.to(device) for target in targets]
            losses, other_vars, loss_dict, log_loss_dict = self.forward_train(img=img, mask=mask, img_metas=img_metas,targets=targets, **kwargs)
            return losses, other_vars, loss_dict, log_loss_dict
        else:

            device = torch.device("cuda")
            targets = [target.to(device) for target in targets]
            output = self.forward_test(img=img, mask=mask, img_metas=img_metas, img_ids=image_id, targets=targets, **kwargs)
            return output

            # get onnx model
            # device = torch.device("cpu")
            # img = img.to(device)
            # self.to(device)
            # # image_path = img_metas[0][0]['filename']
            # # path = '/home2/wr21125091/Anchor3DLane/data/OpenLane/images'
            # # relative_path = os.path.relpath(image_path, path)
            # # base, ext = os.path.splitext(relative_path)
            # # image_id = base
            # # id_path = base + '.txt'
            # # pkl_path = base + '.pkl'
            # # pkl_path = os.path.join('/home2/wr21125091/Anchor3DLane/data/OpenLane/cache_dense/', pkl_path)
            # # with open(pkl_path, 'rb') as f:
            # #     obj = pickle.load(f)
            # # E_inv = np.linalg.inv(obj['gt_camera_extrinsic'])[0:3, :]
            # # gt_project_matrix = np.matmul(obj['gt_camera_intrinsic'], E_inv)
            # # tensor = torch.tensor(gt_project_matrix, dtype=torch.float32)
            # # gt_project_matrix = tensor.unsqueeze(0)
            # # calib_data_path = '/home3/wr21125091/newkitti1(480*320)/calib/'
            # # calib_path = os.path.join(calib_data_path, id_path)
            # intrinsic = np.array([[1.0485574848342735e+03, 0.0, 9.8962676568793063e+02],
            #                       [0.0, 1.0485574848342735e+03, 5.5352172130363283e+02],
            #                       [0.0, 0.0, 1.0]])
            # E = np.array([[1, 0, 0, 0],
            #               [0, 0, 1, 0],
            #               [0, -1, 0, 1.36],
            #               [0, 0, 0, 1]])
            # E_inv = np.linalg.inv(E)[0:3, :]
            # gt_project_matrix = np.matmul(intrinsic, E_inv)
            # tensor = torch.tensor(gt_project_matrix, dtype=torch.float32)
            # gt_project_matrix = tensor.unsqueeze(0)
            # calib_path = "/home1/wr21125091/imagetest/training/calib/000002.txt"
            # calib = Calibration(calib_path, use_right_cam=False)
            # img_w, img_h = (480, 320)
            # pad_size = [0, 0]
            # self.input_height = 320
            # self.input_width = 480
            # self.down_ratio = 4
            # self.output_width = self.input_width // self.down_ratio
            # self.output_height = self.input_height // self.down_ratio
            # self.output_size = [self.output_width, self.output_height]
            # self.max_edge_length = int((self.output_width + self.output_height) * 2)
            # input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
            # edge_indices = self.get_edge_utils((img_w, img_h), pad_size).numpy()
            # input_edge_count = edge_indices.shape[0]
            # input_edge_indices[:edge_indices.shape[0]] = edge_indices
            # input_edge_count = input_edge_count - 1
            # target = ParamsList(image_size=(480, 320), is_train=False)
            # target.add_field("pad_size", pad_size)
            # target.add_field("calib", calib)
            # target.add_field("ori_img", img)
            # target.add_field('edge_len', input_edge_count)
            # target.add_field('edge_indices', input_edge_indices)
            # targets = [target]
            # targets = [target.to(device) for target in targets]
            # output, x = self.forward_test(img=img, mask=mask, img_metas=img_metas, img_ids=image_id, targets=targets,
            #                               gt_project_matrix=gt_project_matrix, **kwargs)
            # return output, x

    @force_fp32()
    def loss(self, output, gt_3dlanes, output_aux=None):
        losses = dict()

        # postprocess
        proposals_list = []
        for proposal, anchor in zip(output['reg_proposals'], output['anchors']):
            proposals_list.append((proposal, anchor))
        anchor_losses = self.lane_loss(proposals_list, gt_3dlanes)
        losses.update(anchor_losses['losses'])
        
        # auxiliary loss
        for iter in range(self.iter_reg):
            proposals_list_aux = []
            for proposal, anchor in zip(output_aux['reg_proposals'][iter], output_aux['anchors'][iter]):
                proposals_list_aux.append((proposal, anchor))
            anchor_losses_aux = self.aux_loss[iter](proposals_list_aux, gt_3dlanes)
            for k, v in anchor_losses_aux['losses'].items():
                if 'loss' in k:
                    losses[k+str(iter)] = v
                
        other_vars = {}
        other_vars['batch_positives'] = anchor_losses['batch_positives']
        other_vars['batch_negatives'] = anchor_losses['batch_negatives']
        return losses, other_vars

    @auto_fp16(apply_to=('img', 'mask', ))
    def forward_train(self, img=None, mask=None, img_metas=None, images=None, targets=None, img_ids=None, gt_3dlanes=None, gt_project_matrix=None, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        gt_project_matrix = gt_project_matrix.squeeze(1)
        output, output_aux, loss_dict, log_loss_dict = self.encoder_decoder(img, mask, gt_project_matrix, targets, **kwargs)
        losses, other_vars = self.loss(output, gt_3dlanes, output_aux)
        return losses, other_vars, loss_dict, log_loss_dict

    def train_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # loss, other_vars, loss_dict, log_loss_dict = self(**data_batch)
        # loss, log_vars = self._parse_losses(loss, other_vars)
        # losses = sum(loss for loss in loss_dict.values())
        # with open('/home2/wr21125091/Anchor3DLane2/output/openlane/detectionloss.txt', 'a') as f:
        #     f.write(str(losses.item()) + "\n")
        # with open('/home2/wr21125091/Anchor3DLane2/output/openlane/laneloss.txt', 'a') as f:
        #     f.write(str(loss.item()) + "\n")
        # with open('/home2/wr21125091/Anchor3DLane2/output/openlane/totalloss.txt', 'a') as f:
        #     f.write(str((loss + losses).item()) + "\n")
        # outputs = dict(
        #     loss=loss + losses,
        #     log_vars=log_vars,
        #     num_samples=data_batch['img'].shape[0])
        # return outputs
        # iter = kwargs['iter']
        # laneloss = kwargs['laneloss']
        # detectionloss = kwargs['detectionloss']
        # meanlaneloss = kwargs['meanlaneloss']
        # meandetectionloss = kwargs['meandetectionloss']
        # if isinstance(meanlaneloss, torch.Tensor):
        #     meanlaneloss = meanlaneloss.tolist()
        # if isinstance(meandetectionloss, torch.Tensor):
        #     meandetectionloss = meandetectionloss.tolist()
        loss, other_vars, loss_dict, log_loss_dict = self(**data_batch)
        loss, log_vars = self._parse_losses(loss, other_vars)
        losses = sum(loss for loss in loss_dict.values())
        # laneloss = laneloss + loss
        # detectionloss = detectionloss + losses.item()
        # with open('/home2/wr21125091/Anchor3DLane2/output/openlane/detectionloss.txt', 'a') as f:
        #     f.write(str(losses.item()) + "\n")
        # with open('/home2/wr21125091/Anchor3DLane2/output/openlane/laneloss.txt', 'a') as f:
        #     f.write(str(loss.item()) + "\n")
        # quotient, remainder = divmod((iter + 1), 950)
        # if remainder == 0:
        #     mean_lane = (laneloss / 950).item()
        #     mean_detection = (detectionloss / 950).item()
        #     meanlaneloss.append(mean_lane)
        #     meandetectionloss.append(mean_detection)
        #     meanlaneloss = torch.tensor(meanlaneloss, dtype=torch.float32, device='cuda:0')
        #     meandetectionloss = torch.tensor(meandetectionloss, dtype=torch.float32, device='cuda:0')
        #     laneloss = torch.tensor([0.0])
        #     detectionloss = torch.tensor([0.0])
        #     del mean_detection
        #     del mean_lane
        #     # print(meanlaneloss)
        #     # print(meandetectionloss)
        # if quotient < 2:
        #     w1 = torch.tensor([1.0]).to('cuda:0')
        #     w2 = torch.tensor([1.0]).to('cuda:0')
        # else:
        #     det_1 = meandetectionloss[quotient - 1]
        #     det_2 = meandetectionloss[quotient - 0]
        #     sudo_det = float(det_2) / float(det_1)
        #     lane_1 = meanlaneloss[quotient - 1]
        #     lane_2 = meanlaneloss[quotient - 0]
        #     sudo_lane = float(lane_2) / float(lane_1)
        #     T = 1.2
        #     K = 2
        #     x = torch.tensor([sudo_det / T, sudo_lane / T])
        #     softmax_x = torch.softmax(x, dim=0)
        #     w1 = softmax_x[0] * K
        #     w2 = softmax_x[1] * K
        #     w1 = w1.to('cuda:0')
        #     w2 = w2.to('cuda:0')
        # with open('/home2/wr21125091/Anchor3DLane2/output/openlane/totalloss.txt', 'a') as f:
        #     f.write(str((w2 * loss + w1 * losses).item()) + "\n")

        outputs = dict(
            loss= loss + losses,
            log_vars=log_vars,
            num_samples=data_batch['img'].shape[0])
        return outputs
        # return outputs, laneloss, detectionloss, meanlaneloss, meandetectionloss

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))

        return outputs

    @staticmethod
    def _parse_losses(losses, other_vars=None):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for var_name, var_value in other_vars.items():
            log_vars[var_name] = var_value
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            # print("log_val_length before reduce:", log_var_length)
            dist.all_reduce(log_var_length)
            # print("log_val_length after reduce:", log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            if isinstance(loss_value, int) or isinstance(loss_value, float):
                continue
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars

    def cuda(self, device=None):
        cuda_self = super().cuda(device)
        cuda_self.anchors = cuda_self.anchors.cuda(device)
        cuda_self.feat_anchors = cuda_self.feat_anchors.cuda(device)
        cuda_self.zs = cuda_self.zs.cuda(device)
        cuda_self.ys = cuda_self.ys.cuda(device)
        cuda_self.xs = cuda_self.xs.cuda(device)
        return cuda_self

    def to(self, *args, **kwargs):
        device_self = super().to(*args, **kwargs)
        device_self.anchors = device_self.anchors.to(*args, **kwargs)
        device_self.feat_anchors = device_self.feat_anchors.to(*args, **kwargs)
        device_self.zs = device_self.zs.to(*args, **kwargs)
        device_self.ys = device_self.ys.to(*args, **kwargs)
        device_self.xs = device_self.xs.to(*args, **kwargs)
        return device_self

    def bilinear_grid_sample(self, im: object, grid: object, align_corners: object = False) -> object:
        n, c, h, w = int(im.shape[0]), int(im.shape[1]), int(im.shape[2]), int(im.shape[3])
        gn, gh, gw, _ = int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2]), int(grid.shape[3])
        x = grid[:, :, :, 0]
        y = grid[:, :, :, 1]
        if align_corners:
            x = ((x + 1) / 2) * (w - 1)
            y = ((y + 1) / 2) * (h - 1)
        else:
            x = ((x + 1) * w - 1) / 2
            y = ((y + 1) * h - 1) / 2
        x = x.view(gn, gh * gw)
        y = y.view(gn, gh * gw)
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1
        wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        wd = ((x - x0) * (y - y0)).unsqueeze(1)
        im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
        padded_h = h + 2
        padded_w = w + 2
        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1
        x0 = torch.where(x0 < 0, torch.tensor(0, device=x0.device, dtype=x0.dtype), x0)
        x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1, device=x0.device, dtype=x0.dtype), x0)
        x1 = torch.where(x1 < 0, torch.tensor(0, device=x0.device, dtype=x0.dtype), x1)
        x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1, device=x0.device, dtype=x0.dtype), x1)
        y0 = torch.where(y0 < 0, torch.tensor(0, device=x0.device, dtype=x0.dtype), y0)
        y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1, device=x0.device, dtype=x0.dtype), y0)
        y1 = torch.where(y1 < 0, torch.tensor(0, device=x0.device, dtype=x0.dtype), y1)
        y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1, device=x0.device, dtype=x0.dtype), y1)
        im_padded = im_padded.view(n, c, (h + 2) * (w + 2))
        x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
        x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
        B, C, _ = im_padded.shape  # B=1, C=64
        _, _, L = x0_y0.shape  # L=44310
        batch_idx = torch.arange(B).view(B, 1, 1).expand(B, C, L)
        channel_idx = torch.arange(C).view(1, C, 1).expand(B, C, L)
        Ia = im_padded[batch_idx, channel_idx, x0_y0]
        Ib = im_padded[batch_idx, channel_idx, x0_y1]
        Ic = im_padded[batch_idx, channel_idx, x1_y0]
        Id = im_padded[batch_idx, channel_idx, x1_y1]
        wa = wa.expand(-1, 64, -1)
        wb = wb.expand(-1, 64, -1)
        wc = wc.expand(-1, 64, -1)
        wd = wd.expand(-1, 64, -1)
        return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

