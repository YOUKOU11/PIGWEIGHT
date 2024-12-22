model = dict(
    type='ImageClassifier',
    backbone=dict(type='MobileNetV3', arch='small'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=4,
        in_channels=576,
        mid_channels=[1024],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='Normal', layer='Linear', mean=0.0, std=0.01, bias=0.0),
        topk=(1, 5)))
dataset_type = 'PigNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Posterize',
            'bits': 4,
            'prob': 0.4
        }, {
            'type': 'Rotate',
            'angle': 30.0,
            'prob': 0.6
        }],
                  [{
                      'type': 'Solarize',
                      'thr': 113.77777777777777,
                      'prob': 0.6
                  }, {
                      'type': 'AutoContrast',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.6
                  }, {
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Solarize',
                      'thr': 142.22222222222223,
                      'prob': 0.2
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Solarize',
                      'thr': 170.66666666666666,
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Posterize',
                      'bits': 6,
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 10.0,
                      'prob': 0.2
                  }, {
                      'type': 'Solarize',
                      'thr': 28.444444444444443,
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.6
                  }, {
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.4
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }, {
                      'type': 'ColorTransform',
                      'magnitude': 0.0,
                      'prob': 0.4
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 30.0,
                      'prob': 0.4
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.0
                  }, {
                      'type': 'Equalize',
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Invert',
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.4,
                      'prob': 0.6
                  }, {
                      'type': 'Contrast',
                      'magnitude': 0.8,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }, {
                      'type': 'ColorTransform',
                      'magnitude': 0.2,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.8,
                      'prob': 0.8
                  }, {
                      'type': 'Solarize',
                      'thr': 56.888888888888886,
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Sharpness',
                      'magnitude': 0.7,
                      'prob': 0.4
                  }, {
                      'type': 'Invert',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Shear',
                      'magnitude': 0.16666666666666666,
                      'prob': 0.6,
                      'direction': 'horizontal'
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.0,
                      'prob': 0.4
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Solarize',
                      'thr': 142.22222222222223,
                      'prob': 0.2
                  }],
                  [{
                      'type': 'Solarize',
                      'thr': 113.77777777777777,
                      'prob': 0.6
                  }, {
                      'type': 'AutoContrast',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Invert',
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.4,
                      'prob': 0.6
                  }, {
                      'type': 'Contrast',
                      'magnitude': 0.8,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }]]),
    dict(
        type='RandomErasing',
        erase_prob=0.2,
        mode='const',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[123.675, 116.28, 103.53]),
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
    dict(type='Resize', size=(256, -1), backend='pillow'),
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
            dict(type='RandomResizedCrop', size=224, backend='pillow'),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type': 'Posterize',
                    'bits': 4,
                    'prob': 0.4
                }, {
                    'type': 'Rotate',
                    'angle': 30.0,
                    'prob': 0.6
                }],
                          [{
                              'type': 'Solarize',
                              'thr': 113.77777777777777,
                              'prob': 0.6
                          }, {
                              'type': 'AutoContrast',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.8
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Posterize',
                              'bits': 5,
                              'prob': 0.6
                          }, {
                              'type': 'Posterize',
                              'bits': 5,
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.4
                          }, {
                              'type': 'Solarize',
                              'thr': 142.22222222222223,
                              'prob': 0.2
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.4
                          }, {
                              'type': 'Rotate',
                              'angle': 26.666666666666668,
                              'prob': 0.8
                          }],
                          [{
                              'type': 'Solarize',
                              'thr': 170.66666666666666,
                              'prob': 0.6
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Posterize',
                              'bits': 6,
                              'prob': 0.8
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 10.0,
                              'prob': 0.2
                          }, {
                              'type': 'Solarize',
                              'thr': 28.444444444444443,
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.6
                          }, {
                              'type': 'Posterize',
                              'bits': 5,
                              'prob': 0.4
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 26.666666666666668,
                              'prob': 0.8
                          }, {
                              'type': 'ColorTransform',
                              'magnitude': 0.0,
                              'prob': 0.4
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 30.0,
                              'prob': 0.4
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.0
                          }, {
                              'type': 'Equalize',
                              'prob': 0.8
                          }],
                          [{
                              'type': 'Invert',
                              'prob': 0.6
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.4,
                              'prob': 0.6
                          }, {
                              'type': 'Contrast',
                              'magnitude': 0.8,
                              'prob': 1.0
                          }],
                          [{
                              'type': 'Rotate',
                              'angle': 26.666666666666668,
                              'prob': 0.8
                          }, {
                              'type': 'ColorTransform',
                              'magnitude': 0.2,
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.8,
                              'prob': 0.8
                          }, {
                              'type': 'Solarize',
                              'thr': 56.888888888888886,
                              'prob': 0.8
                          }],
                          [{
                              'type': 'Sharpness',
                              'magnitude': 0.7,
                              'prob': 0.4
                          }, {
                              'type': 'Invert',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Shear',
                              'magnitude': 0.16666666666666666,
                              'prob': 0.6,
                              'direction': 'horizontal'
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.0,
                              'prob': 0.4
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.4
                          }, {
                              'type': 'Solarize',
                              'thr': 142.22222222222223,
                              'prob': 0.2
                          }],
                          [{
                              'type': 'Solarize',
                              'thr': 113.77777777777777,
                              'prob': 0.6
                          }, {
                              'type': 'AutoContrast',
                              'prob': 0.6
                          }],
                          [{
                              'type': 'Invert',
                              'prob': 0.6
                          }, {
                              'type': 'Equalize',
                              'prob': 1.0
                          }],
                          [{
                              'type': 'ColorTransform',
                              'magnitude': 0.4,
                              'prob': 0.6
                          }, {
                              'type': 'Contrast',
                              'magnitude': 0.8,
                              'prob': 1.0
                          }],
                          [{
                              'type': 'Equalize',
                              'prob': 0.8
                          }, {
                              'type': 'Equalize',
                              'prob': 0.6
                          }]]),
            dict(
                type='RandomErasing',
                erase_prob=0.2,
                mode='const',
                min_area_ratio=0.02,
                max_area_ratio=0.3333333333333333,
                fill_color=[123.675, 116.28, 103.53]),
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
            dict(type='Resize', size=(256, -1), backend='pillow'),
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
        '../data/my_dataset/val_filelist/',
        ann_file='../data/my_dataset/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1), backend='pillow'),
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
optimizer = dict(type='Adam', lr=0.0007, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[5, 30, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=41, hooks=[dict(type='TextLoggerHook')], metric='accuracy precision')
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "../pre_train_mode/mobilenet.pth"
resume_from = None
workflow = [('train', 1)]
policies = [[{
    'type': 'Posterize',
    'bits': 4,
    'prob': 0.4
}, {
    'type': 'Rotate',
    'angle': 30.0,
    'prob': 0.6
}],
            [{
                'type': 'Solarize',
                'thr': 113.77777777777777,
                'prob': 0.6
            }, {
                'type': 'AutoContrast',
                'prob': 0.6
            }],
            [{
                'type': 'Equalize',
                'prob': 0.8
            }, {
                'type': 'Equalize',
                'prob': 0.6
            }],
            [{
                'type': 'Posterize',
                'bits': 5,
                'prob': 0.6
            }, {
                'type': 'Posterize',
                'bits': 5,
                'prob': 0.6
            }],
            [{
                'type': 'Equalize',
                'prob': 0.4
            }, {
                'type': 'Solarize',
                'thr': 142.22222222222223,
                'prob': 0.2
            }],
            [{
                'type': 'Equalize',
                'prob': 0.4
            }, {
                'type': 'Rotate',
                'angle': 26.666666666666668,
                'prob': 0.8
            }],
            [{
                'type': 'Solarize',
                'thr': 170.66666666666666,
                'prob': 0.6
            }, {
                'type': 'Equalize',
                'prob': 0.6
            }],
            [{
                'type': 'Posterize',
                'bits': 6,
                'prob': 0.8
            }, {
                'type': 'Equalize',
                'prob': 1.0
            }],
            [{
                'type': 'Rotate',
                'angle': 10.0,
                'prob': 0.2
            }, {
                'type': 'Solarize',
                'thr': 28.444444444444443,
                'prob': 0.6
            }],
            [{
                'type': 'Equalize',
                'prob': 0.6
            }, {
                'type': 'Posterize',
                'bits': 5,
                'prob': 0.4
            }],
            [{
                'type': 'Rotate',
                'angle': 26.666666666666668,
                'prob': 0.8
            }, {
                'type': 'ColorTransform',
                'magnitude': 0.0,
                'prob': 0.4
            }],
            [{
                'type': 'Rotate',
                'angle': 30.0,
                'prob': 0.4
            }, {
                'type': 'Equalize',
                'prob': 0.6
            }],
            [{
                'type': 'Equalize',
                'prob': 0.0
            }, {
                'type': 'Equalize',
                'prob': 0.8
            }],
            [{
                'type': 'Invert',
                'prob': 0.6
            }, {
                'type': 'Equalize',
                'prob': 1.0
            }],
            [{
                'type': 'ColorTransform',
                'magnitude': 0.4,
                'prob': 0.6
            }, {
                'type': 'Contrast',
                'magnitude': 0.8,
                'prob': 1.0
            }],
            [{
                'type': 'Rotate',
                'angle': 26.666666666666668,
                'prob': 0.8
            }, {
                'type': 'ColorTransform',
                'magnitude': 0.2,
                'prob': 1.0
            }],
            [{
                'type': 'ColorTransform',
                'magnitude': 0.8,
                'prob': 0.8
            }, {
                'type': 'Solarize',
                'thr': 56.888888888888886,
                'prob': 0.8
            }],
            [{
                'type': 'Sharpness',
                'magnitude': 0.7,
                'prob': 0.4
            }, {
                'type': 'Invert',
                'prob': 0.6
            }],
            [{
                'type': 'Shear',
                'magnitude': 0.16666666666666666,
                'prob': 0.6,
                'direction': 'horizontal'
            }, {
                'type': 'Equalize',
                'prob': 1.0
            }],
            [{
                'type': 'ColorTransform',
                'magnitude': 0.0,
                'prob': 0.4
            }, {
                'type': 'Equalize',
                'prob': 0.6
            }],
            [{
                'type': 'Equalize',
                'prob': 0.4
            }, {
                'type': 'Solarize',
                'thr': 142.22222222222223,
                'prob': 0.2
            }],
            [{
                'type': 'Solarize',
                'thr': 113.77777777777777,
                'prob': 0.6
            }, {
                'type': 'AutoContrast',
                'prob': 0.6
            }],
            [{
                'type': 'Invert',
                'prob': 0.6
            }, {
                'type': 'Equalize',
                'prob': 1.0
            }],
            [{
                'type': 'ColorTransform',
                'magnitude': 0.4,
                'prob': 0.6
            }, {
                'type': 'Contrast',
                'magnitude': 0.8,
                'prob': 1.0
            }],
            [{
                'type': 'Equalize',
                'prob': 0.8
            }, {
                'type': 'Equalize',
                'prob': 0.6
            }]]
optimizer = dict(
    type='RMSprop',
    lr=0.064,
    alpha=0.9,
    momentum=0.9,
    eps=0.0316,
    weight_decay=1e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=2, gamma=0.973, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=600)
work_dir = './work_dirs\mobilenet-v3-small_8xb32_in1k'
gpu_ids = [0]
