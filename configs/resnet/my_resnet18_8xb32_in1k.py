import warnings

warnings.filterwarnings("ignore")

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3,),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
            type='VisionTransformerClsHead',
            num_classes=4,
            in_channels=512,
            loss=dict(
                type='LabelSmoothLoss', label_smooth_val=0.1,
                mode='classy_vision'))
    # head=dict(
    #     type='LinearClsHead',
    #     num_classes=6,
    #     in_channels=512,
    #     loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    #     topk=(1, 4))
)
dataset_type = 'PigNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='RandomGrayscale'),
    dict(type='RandomResizedCrop', size=224),
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
    dict(type='Resize', size=(224, -1)),
    dict(type='CenterCrop', crop_size=224),
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
        type='PigNet',
        data_prefix=
        '../data/my_dataset/train_filelist/',
        ann_file='../data/my_dataset/train.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224),
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
        type='PigNet',
        data_prefix=
        '../data/my_dataset/val_filelist/',
        ann_file='../data/my_dataset/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='PigNet',
        data_prefix=
        '../data/my_dataset/test_filelist/',
        ann_file='../data/my_dataset/test.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
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
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[5, 20, 40, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=41, hooks=[dict(type='TextLoggerHook')], metric='accuracy precision')
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "../pre_train_mode/resnet18.pth"
resume_from = None
workflow = [('train', 1), ('val', 1)]
work_dir = './work_dirs/resnet18_8xb32_in1k'
gpu_ids = [0]
