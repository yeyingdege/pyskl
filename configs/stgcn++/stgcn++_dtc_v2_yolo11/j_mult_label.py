multi_label = True
num_classes = 7
clip_len = 100 # default 100
seed = 1234 # 0, 10, 21, 111, 1234

segment_duration = 5.0
overlap_duration = 0.0
# work_dir = f'./work_dirs/stgcn++/stgcn++_dtc_multi-label/j_ml{int(multi_label)}_seed{seed}'
# work_dir = f'./work_dirs/stgcn++/stgcn++_dtc_multi-label-seg3-feb19/j_cl{clip_len}_ml{int(multi_label)}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}'
# work_dir = f'./work_dirs/stgcn++/stgcn++_dtc_multi-label-seg3_linear_prob/j_cl{clip_len}_ml{int(multi_label)}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}'
work_dir = f'./work_dirs/stgcn++/stgcn++_dtc_multi-label-knk1/j_cl{clip_len}_ml{int(multi_label)}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}'

# ann_file = f"data/DTC/multi-label/dtc{num_classes}_ml{int(multi_label)}_seed{seed}.pkl"
# ann_file = f"data/DTC/multi-label-seg3/dtc{num_classes}_ml{int(multi_label)}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}.pkl"
# ann_file = f"data/DTC/multi-label-seg3-feb19/dtc{num_classes}_ml{int(multi_label)}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}.pkl"
ann_file = f"data/DTC/multi-label-knk1/dtc{num_classes}_ml{int(multi_label)}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}_stratified.pkl"

linear_prob = False
# load_from = 'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth'
load_from = 'work_dirs/stgcn++/stgcn++_dtc_multi-label-seg3-feb19/j_cl100_ml1_t5.0_ovlp0.0_seed10/best_mean_f1_epoch_16.pth'
load_from_strict = False
dataset_type = 'PoseDataset'

if multi_label:
    model = dict(
        type='RecognizerGCN',
        backbone=dict(
            type='STGCN',
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn',
            graph_cfg=dict(layout='coco', mode='spatial')),
        cls_head=dict(type='GCNHead', num_classes=num_classes, in_channels=256, loss_cls=dict(type='BCELossWithLogits')),
        test_cfg=dict(average_clips='sigmoid'))
else:
    model = dict(
        type='RecognizerGCN',
        backbone=dict(
            type='STGCN',
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn',
            graph_cfg=dict(layout='coco', mode='spatial')),
        cls_head=dict(type='GCNHead', num_classes=num_classes, in_channels=256))

train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    # dict(type='FormatGCNInputParallel', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
# multi-label setting
if multi_label:
    data = dict(
        videos_per_gpu=16,
        workers_per_gpu=2,
        test_dataloader=dict(videos_per_gpu=1),
        train=dict(
            type='RepeatDataset',
            times=5,
            dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train', multi_class=True, num_classes=num_classes)),
        val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val', multi_class=True, num_classes=num_classes),
        test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val', multi_class=True, num_classes=num_classes))
else:
    data = dict(
        videos_per_gpu=16,
        workers_per_gpu=2,
        test_dataloader=dict(videos_per_gpu=1),
        train=dict(
            type='RepeatDataset',
            times=5,
            dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
        val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
        test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val'))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 10 # default 20
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['f1'], rule='greater', save_best='mean_f1') # 'recall', 'f1', 'mean_f1'
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
