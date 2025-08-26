# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
import json
from ..builder import PIPELINES
import os
from PIL import Image
import cv2
from data.datasets.kitti_utils import Calibration, read_label, approx_proj_center, refresh_attributes, show_heatmap, show_image_with_boxes, show_edge_heatmap
from structures.params_3d import ParamsList
from model.heatmap_coder import (
gaussian_radius,
draw_umich_gaussian,
draw_gaussian_1D,
draw_ellip_gaussian,
draw_umich_gaussian_2D,
)
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.register_module()
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """

        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Transpose(object):
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, order={self.order})'


@PIPELINES.register_module()
class ToDataContainer(object):
    """Convert results to :obj:`mmcv.DataContainer` by given fields.

    Args:
        fields (Sequence[dict]): Each field is a dict like
            ``dict(key='xxx', **kwargs)``. The ``key`` in result will
            be converted to :obj:`mmcv.DataContainer` with ``**kwargs``.
            Default: ``(dict(key='img', stack=True),
            dict(key='gt_semantic_seg'))``.
    """

    def __init__(self,
                 fields=(dict(key='img',
                              stack=True), dict(key='gt_semantic_seg'))):
        self.fields = fields

    def __call__(self, results):
        """Call function to convert data in results to
        :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted to
                :obj:`mmcv.DataContainer`.
        """

        for field in self.fields:
            field = field.copy()
            key = field.pop('key')
            results[key] = DC(results[key], **field)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(fields={self.fields})'


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "gt_semantic_seg".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the bottom/right
            if the batch tensor is larger than this shape.

        - "scale_factor": a float indicating the preprocessing scale

        - "flip": a boolean indicating if image flip transform was used

        - "filename": path to the image file

        - "ori_shape": original shape of the image as a tuple (h, w, c)

        - "pad_shape": image shape after padding

        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: (``filename``, ``ori_filename``, ``ori_shape``,
            ``img_shape``, ``pad_shape``, ``scale_factor``, ``flip``,
            ``flip_direction``, ``img_norm_cfg``)
    """

    def __init__(self,
                 keys,
                 # meta_keys=('filename', 'ori_filename', 'ori_shape',
                 #            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                 #            'flip_direction', 'img_norm_cfg')):
                 meta_keys = ('filename', 'ori_filename', 'ori_shape',
                     'img_shape', 'pad_shape', 'scale_factor',
                     'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:mmcv.DataContainer.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        # data['img_metas'] = DC(img_meta, cpu_only=True)
        results['img_metas'] = DC(img_meta, cpu_only=True)
        origin_path = results['filename']
        front_path = '/root/autodl-tmp/data/OpenLane/images/'
        relative_path = os.path.relpath(origin_path, front_path)
        base, ext = os.path.splitext(relative_path)
        id_path = base + '.txt'
        mode = id_path.split('/')[0]
        label_data_path = '/root/autodl-tmp/newkitti1(480*320)/label/'
        calib_data_path = '/root/autodl-tmp/newkitti1(480*320)/calib/'
        # label_data_path = '/home3/wr21125091/repvf_workspace/withoutoutside/label/'
        # calib_data_path = '/home3/wr21125091/repvf_workspace/withoutoutside/calib/'
        label_path = os.path.join(label_data_path, id_path)
        calib_path = os.path.join(calib_data_path, id_path)
        self.input_height = 320
        self.input_width = 480
        self.down_ratio = 4
        self.output_width = self.input_width // self.down_ratio
        self.output_height = self.input_height // self.down_ratio
        self.output_size = [self.output_width, self.output_height]
        self.max_edge_length = int((self.output_width + self.output_height) * 2)
        self.num_classes = 3
        self.max_objs = 40
        self.multibin_size = 4
        self.consider_outside_objs = True
        self.filter_annos = True
        self.filter_params = [0.9, 20]
        self.heatmap_center = '3D'
        self.adjust_edge_heatmap = True
        self.enable_edge_fusion = True
        self.orientation_method = 'multi-bin'
        self.classes = ("Car", "Pedestrian", "Cyclist")
        self.use_modify_keypoint_visible = True
        self.multibin_size = 4
        self.edge_heatmap_ratio = 0.5
        PI = np.pi
        self.alpha_centers = np.array([0, PI / 2, PI, - PI / 2])
        self.proj_center_mode = 'intersect'
        if mode == 'training':
            self.is_train = True
        else:
            self.is_train = False
        img = Image.open(origin_path).convert('RGB')
        size = (480, 320)
        img = img.resize(size, Image.BILINEAR)
        calib = Calibration(calib_path, use_right_cam=False)
        objs = read_label(label_path)
        objs = self.filtrate_objects(objs)
        img_before_aug_pad = np.array(img).copy()
        img_w, img_h = img.size
        img, pad_size = self.pad_image(img)
        ori_img = np.array(img).copy()
        # the boundaries of the image after padding
        x_min, y_min = int(np.ceil(pad_size[0] / self.down_ratio)), int(np.ceil(pad_size[1] / self.down_ratio))
        x_max, y_max = (pad_size[0] + img_w - 1) // self.down_ratio, (pad_size[1] + img_h - 1) // self.down_ratio
        if self.enable_edge_fusion:
            # generate edge_indices for the edge fusion module
            input_edge_indices = np.zeros([self.max_edge_length, 2], dtype=np.int64)
            edge_indices = self.get_edge_utils((img_w, img_h), pad_size).numpy()
            input_edge_count = edge_indices.shape[0]
            input_edge_indices[:edge_indices.shape[0]] = edge_indices
            input_edge_count = input_edge_count - 1  # explain ?

        # heatmap
        heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        ellip_heat_map = np.zeros([self.num_classes, self.output_height, self.output_width], dtype=np.float32)
        # classification
        cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        target_centers = np.zeros([self.max_objs, 2], dtype=np.int32)
        # 2d bounding boxes
        gt_bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        bboxes = np.zeros([self.max_objs, 4], dtype=np.float32)
        # keypoints: 2d coordinates and visible(0/1)
        keypoints = np.zeros([self.max_objs, 10, 3], dtype=np.float32)
        keypoints_depth_mask = np.zeros([self.max_objs, 3],
                                            dtype=np.float32)  # whether the depths computed from three groups of keypoints are valid
        # 3d dimension
        dimensions = np.zeros([self.max_objs, 3], dtype=np.float32)
        # 3d location
        locations = np.zeros([self.max_objs, 3], dtype=np.float32)
        # rotation y
        rotys = np.zeros([self.max_objs], dtype=np.float32)
        # alpha (local orientation)
        alphas = np.zeros([self.max_objs], dtype=np.float32)
        # offsets from center to expected_center
        offset_3D = np.zeros([self.max_objs, 2], dtype=np.float32)

        # occlusion and truncation
        occlusions = np.zeros(self.max_objs)
        truncations = np.zeros(self.max_objs)

        if self.orientation_method == 'head-axis':
            orientations = np.zeros([self.max_objs, 3], dtype=np.float32)
        else:
            orientations = np.zeros([self.max_objs, self.multibin_size * 2],
                                        dtype=np.float32)  # multi-bin loss: 2 cls + 2 offset

        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)  # regression mask
        trunc_mask = np.zeros([self.max_objs], dtype=np.uint8)  # outside object mask
        reg_weight = np.zeros([self.max_objs], dtype=np.float32)  # regression weight
        TYPE_ID_CONVERSION = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Van': -4,
            'Truck': -4,
            'Person_sitting': -2,
            'Tram': -99,
            'Misc': -99,
            'DontCare': -1,
        }
        for i, obj in enumerate(objs):
            cls = obj.type
            cls_id = TYPE_ID_CONVERSION[cls]
            if cls_id < 0: continue
            float_occlusion = float(
                obj.occlusion)  # 0 for normal, 0.33 for partially, 0.66 for largely, 1 for unknown (mostly very far and small objs)
            float_truncation = obj.truncation  # 0 ~ 1 and stands for truncation level

            # bottom centers ==> 3D centers
            locs = obj.t.copy()
            locs[1] = locs[1] - obj.h / 2
            if locs[-1] <= 0: continue  # objects which are behind the image

            # generate 8 corners of 3d bbox
            corners_3d = obj.generate_corners3d()
            corners_2d, _ = calib.project_rect_to_image(corners_3d)
            projected_box2d = np.array([corners_2d[:, 0].min(), corners_2d[:, 1].min(),
                                        corners_2d[:, 0].max(), corners_2d[:, 1].max()])

            if projected_box2d[0] >= 0 and projected_box2d[1] >= 0 and \
                    projected_box2d[2] <= img_w - 1 and projected_box2d[3] <= img_h - 1:
                box2d = projected_box2d.copy()
            else:
                box2d = obj.box2d.copy()

            # filter some unreasonable annotations
            if self.filter_annos:
                if float_truncation >= self.filter_params[0] and (box2d[2:] - box2d[:2]).min() <= self.filter_params[1]: continue

            # project 3d location to the image plane
            proj_center, depth = calib.project_rect_to_image(locs.reshape(-1, 3))
            proj_center = proj_center[0]

            # generate approximate projected center when it is outside the image
            proj_inside_img = (0 <= proj_center[0] <= img_w - 1) & (0 <= proj_center[1] <= img_h - 1)
            approx_center = False
            if not proj_inside_img:
                if self.consider_outside_objs:
                    approx_center = True
                    center_2d = (box2d[:2] + box2d[2:]) / 2
                    center_2d_x, center_2d_y = center_2d
                    if (center_2d_x >= 0) & (center_2d_y >= 0) & (center_2d_x <= img_w - 1) & (
                            center_2d_y <= img_h - 1):
                        target_proj_center, edge_index = approx_proj_center(proj_center, center_2d.reshape(1, 2),
                                                                            (img_w, img_h))
                    else:
                        continue
                    # if self.proj_center_mode == 'intersect':
                    #     target_proj_center, edge_index = approx_proj_center(proj_center, center_2d.reshape(1, 2),
                    #                                                         (img_w, img_h))
                    # else:
                    #     raise NotImplementedError
                else:
                    continue
            else:
                target_proj_center = proj_center.copy()

            # 10 keypoints
            bot_top_centers = np.stack((corners_3d[:4].mean(axis=0), corners_3d[4:].mean(axis=0)), axis=0)
            keypoints_3D = np.concatenate((corners_3d, bot_top_centers), axis=0)
            keypoints_2D, _ = calib.project_rect_to_image(keypoints_3D)

            # keypoints mask: keypoint must be inside the image and in front of the camera
            keypoints_x_visible = (keypoints_2D[:, 0] >= 0) & (keypoints_2D[:, 0] <= img_w - 1)
            keypoints_y_visible = (keypoints_2D[:, 1] >= 0) & (keypoints_2D[:, 1] <= img_h - 1)
            keypoints_z_visible = (keypoints_3D[:, -1] > 0)

            # xyz visible
            keypoints_visible = keypoints_x_visible & keypoints_y_visible & keypoints_z_visible
            # center, diag-02, diag-13
            keypoints_depth_valid = np.stack((keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(),
                                              keypoints_visible[[1, 3, 5, 7]].all()))

            if self.use_modify_keypoint_visible:
                keypoints_visible = np.append(np.tile(keypoints_visible[:4] | keypoints_visible[4:8], 2),
                                              np.tile(keypoints_visible[8] | keypoints_visible[9], 2))
                keypoints_depth_valid = np.stack((
                                                 keypoints_visible[[8, 9]].all(), keypoints_visible[[0, 2, 4, 6]].all(),
                                                 keypoints_visible[[1, 3, 5, 7]].all()))

                keypoints_visible = keypoints_visible.astype(np.float32)
                keypoints_depth_valid = keypoints_depth_valid.astype(np.float32)

            # downsample bboxes, points to the scale of the extracted feature map (stride = 4)
            keypoints_2D = (keypoints_2D + pad_size.reshape(1, 2)) / self.down_ratio
            target_proj_center = (target_proj_center + pad_size) / self.down_ratio
            proj_center = (proj_center + pad_size) / self.down_ratio

            box2d[0::2] += pad_size[0]
            box2d[1::2] += pad_size[1]
            box2d /= self.down_ratio
            # 2d bbox center and size
            bbox_center = (box2d[:2] + box2d[2:]) / 2
            bbox_dim = box2d[2:] - box2d[:2]

            # target_center: the point to represent the object in the downsampled feature map
            if self.heatmap_center == '2D':
                target_center = bbox_center.round().astype(np.int)
            else:
                target_center = target_proj_center.round().astype(np.int)

            # clip to the boundary
            target_center[0] = np.clip(target_center[0], x_min, x_max)
            target_center[1] = np.clip(target_center[1], y_min, y_max)

            pred_2D = True  # In fact, there are some wrong annotations where the target center is outside the box2d
            if not (target_center[0] >= box2d[0] and target_center[1] >= box2d[1] and target_center[0] <= box2d[2] and
                    target_center[1] <= box2d[3]):
                pred_2D = False

            if (bbox_dim > 0).all() and (0 <= target_center[0] <= self.output_width - 1) and (
                    0 <= target_center[1] <= self.output_height - 1):
                rot_y = obj.ry
                alpha = obj.alpha
                # generating heatmap
                if self.adjust_edge_heatmap and approx_center:
                    # for outside objects, generate 1-dimensional heatmap
                    bbox_width = min(target_center[0] - box2d[0], box2d[2] - target_center[0])
                    bbox_height = min(target_center[1] - box2d[1], box2d[3] - target_center[1])
                    radius_x, radius_y = bbox_width * self.edge_heatmap_ratio, bbox_height * self.edge_heatmap_ratio
                    radius_x, radius_y = max(0, int(radius_x)), max(0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    heat_map[cls_id] = draw_umich_gaussian_2D(heat_map[cls_id], target_center, radius_x, radius_y)
                else:
                    # for inside objects, generate circular heatmap
                    radius = gaussian_radius(bbox_dim[1], bbox_dim[0])
                    radius = max(0, int(radius))
                    heat_map[cls_id] = draw_umich_gaussian(heat_map[cls_id], target_center, radius)
            else:
                continue
            if 0 <= i < len(cls_ids):
                cls_ids[i] = cls_id
                target_centers[i] = target_center
                # offset due to quantization for inside objects or offset from the interesection to the projected 3D center for outside objects
                offset_3D[i] = proj_center - target_center

                # 2D bboxes
                gt_bboxes[i] = obj.box2d.copy()  # for visualization
                if pred_2D: bboxes[i] = box2d

                # local coordinates for keypoints
                keypoints[i] = np.concatenate((keypoints_2D - target_center.reshape(1, -1), keypoints_visible[:, np.newaxis]), axis=1)
                keypoints_depth_mask[i] = keypoints_depth_valid

                dimensions[i] = np.array([obj.l, obj.h, obj.w])
                locations[i] = locs
                rotys[i] = rot_y
                alphas[i] = alpha

                orientations[i] = self.encode_alpha_multibin(alpha, num_bin=self.multibin_size)

                reg_mask[i] = 1
                reg_weight[i] = 1  # all objects are of the same weights (for now)
                trunc_mask[i] = int(approx_center)  # whether the center is truncated and therefore approximate
                occlusions[i] = float_occlusion
                truncations[i] = float_truncation
            else:
                continue
            # visualization
            # img3 = show_image_with_boxes(img, cls_ids, target_centers, bboxes.copy(), keypoints, reg_mask,
            # 							offset_3D, self.down_ratio, pad_size, orientations, vis=True)
            # show_heatmap(img, heat_map, index=original_idx)

        target = ParamsList(image_size=img.size, is_train=self.is_train)
        target.add_field("cls_ids", cls_ids)
        target.add_field("target_centers", target_centers)
        target.add_field("keypoints", keypoints)
        target.add_field("keypoints_depth_mask", keypoints_depth_mask)
        target.add_field("dimensions", dimensions)
        target.add_field("locations", locations)
        target.add_field("calib", calib)
        target.add_field("reg_mask", reg_mask)
        target.add_field("reg_weight", reg_weight)
        target.add_field("offset_3D", offset_3D)
        target.add_field("2d_bboxes", bboxes)
        target.add_field("pad_size", pad_size)
        target.add_field("ori_img", ori_img)
        target.add_field("rotys", rotys)
        target.add_field("trunc_mask", trunc_mask)
        target.add_field("alphas", alphas)
        target.add_field("orientations", orientations)
        target.add_field("hm", heat_map)
        target.add_field("gt_bboxes", gt_bboxes)  # for validation visualization
        target.add_field("occlusions", occlusions)
        target.add_field("truncations", truncations)

        if self.enable_edge_fusion:
            target.add_field('edge_len', input_edge_count)
            target.add_field('edge_indices', input_edge_indices)

        results['targets'] = target
        results['image_id'] = base
        for key in self.keys:
            data[key] = results[key]
        return data

    def filtrate_objects(self, obj_list):
        """
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        valid_obj_list = []
        for obj in obj_list:
            if obj.type not in type_whitelist:
                continue

            valid_obj_list.append(obj)

        return valid_obj_list
    def encode_alpha_multibin(self, alpha, num_bin=2, margin=1 / 6):
        # encode alpha (-PI ~ PI) to 2 classes and 1 regression
        encode_alpha = np.zeros(num_bin * 2)
        bin_size = 2 * np.pi / num_bin  # pi
        margin_size = bin_size * margin  # pi / 6

        bin_centers = self.alpha_centers
        range_size = bin_size / 2 + margin_size

        offsets = alpha - bin_centers
        offsets[offsets > np.pi] = offsets[offsets > np.pi] - 2 * np.pi
        offsets[offsets < -np.pi] = offsets[offsets < -np.pi] + 2 * np.pi

        for i in range(num_bin):
            offset = offsets[i]
            if abs(offset) < range_size:
                encode_alpha[i] = 1
                encode_alpha[i + num_bin] = offset

        return encode_alpha
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
    def pad_image(self, image):
        img = np.array(image)
        h, w, c = img.shape
        ret_img = np.zeros((self.input_height, self.input_width, c))
        pad_y = (self.input_height - h) // 2
        pad_x = (self.input_width - w) // 2

        ret_img[pad_y: pad_y + h, pad_x: pad_x + w] = img
        pad_size = np.array([pad_x, pad_y])

        return Image.fromarray(ret_img.astype(np.uint8)), pad_size
    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'
