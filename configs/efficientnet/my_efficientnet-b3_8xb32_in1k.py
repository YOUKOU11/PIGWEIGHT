import warnings
warnings.filterwarnings("ignore")

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b3'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1536,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'PigNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=300,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='CenterCrop',
        crop_size=300,
        efficientnet_style=True,
        interpolation='bicubic'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='D:/Python项目/classification-master/data/my_dataset/train_filelist/',
        ann_file='D:/Python项目/classification-master/data/my_dataset/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type=dataset_type,
        data_prefix='D:/Python项目/classification-master/data/my_dataset/val_filelist/',
        ann_file='D:/Python项目/classification-master/data/my_dataset/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type=dataset_type,
        data_prefix='D:/Python项目/classification-master/data/my_dataset/test_filelist/',
        ann_file='D:/Python项目/classification-master/data/my_dataset/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='CenterCrop',
                crop_size=300,
                efficientnet_style=True,
                interpolation='bicubic'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, metric='accuracy')
# optimizer = dict(type='SGD', lr=0.0007, momentum=0.8)
optimizer = dict(type='Adam', lr=0.00007, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[60])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=1)
log_config = dict(interval=40, hooks=[dict(type='TextLoggerHook')], metric='accuracy precision')
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "D:/Python项目/classification-master/pre_train_mode/efficientnet-b3_1.pth"
resume_from = None
workflow = [('train', 1), ('val', 1)]
work_dir = './work_dirs/efficientnet-b3_8xb32_in1k'
gpu_ids = [0]