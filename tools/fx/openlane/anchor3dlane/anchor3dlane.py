feat_y_steps = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100]
anchor_y_steps = [
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100
]
anchor_len = 20
dataset_type = 'OpenlaneDataset'
data_root = '/home2/wr21125091/Anchor3DLane/data/OpenLane'
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
input_size = (320, 480)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(480, 320), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='MaskGenerate', input_size=(320, 480)),
    dict(type='LaneFormat'),
    dict(
        type='Collect',
        keys=[
            'img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix', 'mask',
            'targets', 'image_id'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(480, 320), keep_ratio=False),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='MaskGenerate', input_size=(320, 480)),
    dict(type='LaneFormat'),
    dict(
        type='Collect',
        keys=[
            'img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix', 'mask',
            'targets', 'image_id'
        ])
]
dataset_config = dict(max_lanes=25, input_size=(320, 480))
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='OpenlaneDataset',
        data_root='/home2/wr21125091/Anchor3DLane/data/OpenLane',
        data_list='training.txt',
        dataset_config=dict(max_lanes=25, input_size=(320, 480)),
        y_steps=[
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
            90, 95, 100
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(480, 320), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='MaskGenerate', input_size=(320, 480)),
            dict(type='LaneFormat'),
            dict(
                type='Collect',
                keys=[
                    'img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix',
                    'mask', 'targets', 'image_id'
                ])
        ]),
    test=dict(
        type='OpenlaneDataset',
        data_root='/home2/wr21125091/Anchor3DLane/data/OpenLane',
        y_steps=[
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
            90, 95, 100
        ],
        data_list='validation.txt',
        dataset_config=dict(max_lanes=25, input_size=(320, 480)),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', img_scale=(480, 320), keep_ratio=False),
            dict(
                type='Normalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                to_rgb=True),
            dict(type='MaskGenerate', input_size=(320, 480)),
            dict(type='LaneFormat'),
            dict(
                type='Collect',
                keys=[
                    'img', 'img_metas', 'gt_3dlanes', 'gt_project_matrix',
                    'mask', 'targets', 'image_id'
                ])
        ]))
model = dict(
    type='Anchor3DLane',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        with_cp=False,
        style='pytorch'),
    pretrained='/home2/wr21125091/Anchor3DLane2/resnet18_v1c-b5776b93.pth',
    y_steps=[
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
        95, 100
    ],
    feat_y_steps=[5, 10, 15, 20, 30, 40, 50, 60, 80, 100],
    anchor_cfg=dict(
        pitches=[5, 2, 1, 0, -1, -2, -5],
        yaws=[
            30, 20, 15, 10, 7, 5, 3, 1, 0, -1, -3, -5, -7, -10, -15, -20, -30
        ],
        num_x=45,
        distances=[3]),
    db_cfg=dict(
        org_h=1280,
        org_w=1920,
        resize_h=320,
        resize_w=480,
        ipm_h=208,
        ipm_w=128,
        pitch=3,
        cam_height=1.55,
        crop_y=0,
        K=[[2015.0, 0.0, 960.0], [0.0, 2015.0, 540.0], [0.0, 0.0, 1.0]],
        top_view_region=[[-10, 103], [10, 103], [-10, 3], [10, 3]],
        max_2dpoints=10),
    attn_dim=64,
    feat_size=(80, 120),
    num_category=21,
    loss_lane=dict(
        type='LaneLoss',
        loss_weights=dict(
            cls_loss=1, reg_losses_x=1, reg_losses_z=1, reg_losses_vis=1),
        assign_cfg=dict(
            type='TopkAssigner',
            pos_k=3,
            neg_k=450,
            anchor_len=20,
            metric='Euclidean'),
        anchor_len=20,
        anchor_steps=[
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,
            90, 95, 100
        ]),
    train_cfg=dict(nms_thres=0, conf_threshold=0),
    test_cfg=dict(
        nms_thres=2,
        conf_threshold=0.2,
        test_conf=0.5,
        refine_vis=True,
        vis_thresh=0.5))
data_shuffle = True
optimizer = dict(type='AdamW', lr=0.0003, weight_decay=1e-05)
optimizer_config = dict()
lr_config = dict(policy='step', step=[76000, 85500], gamma=0.1, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=95000)
checkpoint_config = dict(by_epoch=False, interval=9500)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 100000000)]
cudnn_benchmark = True
work_dir = 'fx/openlane/anchor3dlane'
gpu_ids = [0]
auto_resume = False
