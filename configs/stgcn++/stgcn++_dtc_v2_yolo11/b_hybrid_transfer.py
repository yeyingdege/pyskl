"""
Hybrid Transfer Learning Config for DTC Dataset
Combines the BEST of both approaches:

From j_improved.py (Training from Scratch):
✅ Strong regularization: dropout=0.7, label_smoothing=0.1
✅ Optimal hyperparameters: lr=0.005, 40 epochs
✅ Data repetition: 5x

From Transfer Learning:
✅ Pretrained NTU RGB+D weights (initialization)
✅ Layer-wise learning rates (discriminative fine-tuning)
✅ Gradual learning rate warmup

Hypothesis: Transfer learning + j_improved's winning formula = 74-75% accuracy
           (Better than simple transfer: 72.24%, potentially better than from-scratch: 73.00%)
"""

multi_label = False
num_classes = 7

# === MODEL CONFIGURATION ===
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',      # ✅ From j_improved: Learn adaptive graphs
        gcn_with_res=True,        # ✅ From j_improved: Residual connections
        tcn_type='mstcn',         # ✅ From j_improved: Multi-scale temporal
        graph_cfg=dict(layout='coco', mode='spatial')
    ),
    cls_head=dict(
        type='GCNHead',
        dropout=0.7,              # ✅ From j_improved: STRONG regularization
        num_classes=num_classes,
        in_channels=256,
        # loss_cls=dict(
        #     type='LabelSmoothLoss',
        #     label_smooth_val=0.1  # ✅ From j_improved: Label smoothing
        # )
    )
)

# === TRANSFER LEARNING: Load Pretrained Weights ===
# Initialize from NTU RGB+D pretrained model
#load_from = 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/b.pth'
load_from = 'https://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/b.pth'

# Note: Classification head will be randomly initialized 
# But backbone will use pretrained weights as initialization

# === DATASET CONFIGURATION ===
dataset_type = 'PoseDataset'
balanced_dataset_type = 'BalancedPoseDataset'
ann_file = 'data/DTC/dtc7.pkl'
clip_len = 100 # 120

# ✅ From j_improved: Same data pipeline
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='RandomRot'),          # Add rotation augmentation
    dict(type='RandomScale', scale=0.1),  # Add scale augmentation

    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='UniformSample', clip_len=clip_len, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

# ✅ From j_improved: Class balancing with RepeatDataset
data = dict(
    videos_per_gpu=24,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,  # ✅ From j_improved: 5x data repetition
        #dataset=dict(
        #    type=balanced_dataset_type,
        #     ann_file=ann_file,
        #     pipeline=train_pipeline,
        #     split='train',
        #     num_classes=num_classes,
            # sample_by_class=True,
            # ✅ From j_improved: Class balancing weights
            # 1.0, 5.4, 3.6, 4.3, 2.1, 4.3, 3.3
            # 3.0, 16.2, 10.8, 12.9, 6.3, 12.9, 9.9
        #     class_prob=[1.0, 5.4, 3.6, 4.3, 2.1, 4.3, 3.3])
        dataset=dict(
                type=dataset_type, 
                ann_file=ann_file, 
                pipeline=train_pipeline, 
                split='train', 
                num_classes=num_classes)
        
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val'
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='val'
    )
)

# === HYBRID OPTIMIZATION STRATEGY ===
# Combines j_improved's LR with transfer learning's layer-wise approach

# NEW: Discriminative Learning Rates (Layer-wise Fine-tuning)
# Different LR for different layers based on their transferability
optimizer = dict(
    type='SGD',
    lr=0.0006,  # ✅ From j_improved: Base LR (works well for DTC)
    momentum=0.9,
    weight_decay=0.0005,  # ✅ From j_improved: L2 regularization
    nesterov=True,

    # NEW: Layer-wise learning rate multipliers
    # Idea: Early layers transfer well (low LR), late layers need adaptation (higher LR)
    paramwise_cfg=dict(
        custom_keys={
            # Early layers (1-3): Transfer well, use very low LR
            # These learn low-level features (edges, joints) that are universal
           'backbone.gcn1': dict(lr_mult=0.1),   # 0.005 * 0.1 = 0.0005
            'backbone.gcn2': dict(lr_mult=0.1),
            'backbone.gcn3': dict(lr_mult=0.1),

            # Middle layers (4-7): Need moderate adaptation
            # These learn temporal patterns that may differ between NTU and DTC
            'backbone.gcn4': dict(lr_mult=0.5),   # 0.005 * 0.5 = 0.0025
            'backbone.gcn5': dict(lr_mult=0.5),
            'backbone.gcn6': dict(lr_mult=0.5),
            'backbone.gcn7': dict(lr_mult=0.5),

            # Late layers (8-10): Need significant adaptation
            # These learn high-level action semantics specific to the task
           'backbone.gcn8': dict(lr_mult=1.0),   # 0.005 * 1.0 = 0.005 (full LR)
            'backbone.gcn9': dict(lr_mult=1.0),
            'backbone.gcn10': dict(lr_mult=1.0),

            # Classification head: Randomly initialized, needs highest LR
            'cls_head': dict(lr_mult=2.0)         # 0.005 * 2.0 = 0.01
        }
    )
)

optimizer_config = dict(grad_clip=None)

# ✅ From j_improved: CosineAnnealing with gradual LR decay
# NEW: Add warmup for transfer learning stability
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    by_epoch=False,

    # NEW: Warmup for transfer learning
    # Gradually increase LR in first 500 iterations to avoid disrupting pretrained weights
    warmup='linear',
    warmup_iters=500,   # ~1 epoch worth
    warmup_ratio=0.1    # Start at 10% of initial LR
)

# ✅ From j_improved: 40 epochs total
# Extended slightly for transfer learning
total_epochs = 20

# ✅ From j_improved: Save best checkpoint
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy'],
    save_best='top1_acc',
    rule='greater'
)

log_config = dict(

    interval=100,
    hooks=[dict(type='TextLoggerHook')]
)

# Set random seed for reproducibility
seed = 42

# Runtime settings
log_level = 'INFO'
work_dir = './work_dirs/stgcn++/stgcn++_dtc_v2_yolo11/b_hybrid_transfer2'

# === EXPECTED RESULTS ===
# Hypothesis:
# - Pretrained initialization: Faster convergence in early epochs
# - Strong regularization (dropout 0.7 + label smoothing): Prevents overfitting
# - Layer-wise LR: Preserves transferable features, adapts task-specific layers
# - Class balancing: Equal learning across all 7 classes
#
# Expected accuracy: 74-75%
# - Better than simple transfer (72.24%)
# - Potentially better than from-scratch (73.00%)
# - Not as good as ensemble (76.81%), but best single-stream approach
#
# Training characteristics:
# - Epochs 1-5:   Fast initial learning (pretrained boost)
# - Epochs 6-20:  Steady improvement (adaptation)
# - Epochs 21-35: Gradual refinement (strong regularization prevents overfitting)
# - Epochs 36-45: Final convergence
