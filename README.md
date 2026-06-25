# PYSKL

> Note: This repo is currently not maintained by the developer. Feel free to create forks and develop based on this piece of codes.


PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

This repo is the official implementation of [PoseConv3D](https://arxiv.org/abs/2104.13586) and [STGCN++](https://github.com/kennymckormick/pyskl/tree/main/configs/stgcn%2B%2B).

<div id="wrapper" align="center">
<figure>
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="520px">&emsp;
  <img src="https://user-images.githubusercontent.com/34324155/218010909-ccfc89f0-9ed4-4b04-b38d-af7ffe49d2cd.gif" width="290px"><br>
  <p style="font-size:1.2vw;">Left: Skeleton-base Action Recognition Results on NTU-RGB+D-120; Right: CPU Realtime Skeleton-base Gesture Recognition Results</p>
</figure>
</div>

## Supported Algorithms

- [x] [DG-STGCN (Arxiv)](https://arxiv.org/abs/2210.05895) [[MODELZOO](/configs/dgstgcn/README.md)]
- [x] [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) [[MODELZOO](/configs/stgcn/README.md)]
- [x] [ST-GCN++ (ACMMM 2022)](https://arxiv.org/abs/2205.09443) [[MODELZOO](/configs/stgcn++/README.md)]
- [x] [PoseConv3D (CVPR 2022 Oral)](https://arxiv.org/abs/2104.13586) [[MODELZOO](/configs/posec3d/README.md)]
- [x] [AAGCN (TIP)](https://arxiv.org/abs/1912.06971) [[MODELZOO](/configs/aagcn/README.md)]
- [x] [MS-G3D (CVPR 2020 Oral)](https://arxiv.org/abs/2003.14111) [[MODELZOO](/configs/msg3d/README.md)]
- [x] [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213) [[MODELZOO](/configs/ctrgcn/README.md)]

## Supported Skeleton Datasets

- [x] [NTURGB+D (CVPR 2016)](https://arxiv.org/abs/1604.02808) and [NTURGB+D 120 (TPAMI 2019)](https://arxiv.org/abs/1905.04757)
- [x] [Kinetics 400 (CVPR 2017)](https://arxiv.org/abs/1705.06950)
- [x] [UCF101 (ArXiv 2012)](https://arxiv.org/pdf/1212.0402.pdf)
- [x] [HMDB51 (ICCV 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6126543)
- [x] [FineGYM (CVPR 2020)](https://arxiv.org/abs/2004.06704)
- [x] [Diving48 (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Installation Python 3.10 (recommended)
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
conda env create -f pyskl_310.yaml
conda activate pyskl
# comment the line 813 raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
# in ~/miniforge3/envs/pyskl_310/lib/python3.10/site-packages/torch/utils/cpp_extension.py
conda env update -n pyskl_310 -f pyskl_310.yaml
pip install -e .
```

## Demo

Check [demo.md](/demo/demo.md).

## Data Preparation

We provide HRNet 2D skeletons for every dataset we support and Kinect 3D skeletons for NTURGB+D/NTURGB+D 120. To prepare skeleton annotations:

1. Use pre-processed skeleton annotations: processed pickle files for supported datasets are described in [Data Doc](/tools/data/README.md).
2. For NTURGB+D 3D skeletons, download official annotations from https://github.com/shahroudy/NTURGB-D and run [tools/data/ntu_preproc.py](/tools/data/ntu_preproc.py) to generate `ntu60_3danno.pkl` and `ntu120_3danno.pkl`.
3. For custom RGB datasets, extract 2D poses first (for example, see [diving48_example](/examples/extract_diving48_skeleton/diving48_example.ipynb)).

### DTC multi-label preprocessing (tools/data/dtc_preproc_v2.py)

The script [tools/data/dtc_preproc_v2.py](/tools/data/dtc_preproc_v2.py) converts per-video JSON pose annotations into a PySKL training pickle with train/val splits.

Expected inputs:

- `--annotation_dir`: directory of JSON annotation files (default: `data/DTC/annotations_1.0_knk_1`).
- `--label_map`: label text file used to map class names to IDs (default: `tools/data/label_map/dtc7.txt`).
- Each JSON file should contain per-person pose/keypoint/action annotations compatible with this script.

What the script does:

- Builds a stratified train/val split at file level (`--train_ratio`, default `0.6`) using label presence.
- Generates multi-label temporal segments per person:
  - Train split uses action-aligned windows.
  - Val split uses fixed window/stride segmentation.
- Filters segments shorter than `--frame_thres` (default `30`) and applies minimum action-overlap frames with `--min_action_frames` (default `15`).
- Saves a pickle with:
  - `split`: `train`/`val` sample IDs.
  - `annotations`: pose samples (`keypoint`, `keypoint_score`, `label`, `total_frames`, etc.).

Example:

```shell
python tools/data/dtc_preproc_v2.py \
  --annotation_dir data/DTC/annotations_1.0_knk_1 \
  --label_map tools/data/label_map/dtc7.txt \
  --out_dir data/DTC/multi-label-knk1 \
  --segment_duration 5.0 \
  --overlap_duration 0.0 \
  --frame_thres 30 \
  --min_action_frames 15 \
  --train_ratio 0.6 \
  --seed 0
```

Output file pattern:

```text
{out_dir}/dtc{num_labels}_ml{0|1}_t{segment_duration}_ovlp{overlap_duration}_seed{seed}_stratified.pkl
```

Example output path with defaults:

```text
data/DTC/multi-label-knk1/dtc7_ml1_t5.0_ovlp0.0_seed0_stratified.pkl
```

## Config: STGCN++ DTC v2 (YOLO11, multi-label)

Reference config:

`configs/stgcn++/stgcn++_dtc_v2_yolo11/j_mult_label.py`

Key configuration values used in this setup:

- `multi_label=True`
- `num_classes=7`
- `clip_len=100`
- `seed=0`
- `segment_duration=5.0`
- `overlap_duration=0.0`
- `dataset_type='PoseDataset'`
- `ann_file=data/DTC/multi-label-knk1/dtc7_ml1_t5.0_ovlp0.0_seed0_stratified.pkl`
- `work_dir=./work_dirs/stgcn++/stgcn++_dtc_multi-label-knk1/j_cl100_ml1_t5.0_ovlp0.0_seed0`

Model and training details:

- Model: `RecognizerGCN` with `STGCN` backbone (`gcn_adaptive='init'`, `gcn_with_res=True`, `tcn_type='mstcn'`, `graph_cfg(layout='coco', mode='spatial')`)
- Head: `GCNHead(in_channels=256, num_classes=7)`
- Multi-label loss: `BCELossWithLogits`
- Test-time clip aggregation: `average_clips='sigmoid'`
- Data: `RepeatDataset(times=5)` for training, `multi_class=True`, `num_classes=7`
- Batch/loader: `videos_per_gpu=16`, `workers_per_gpu=2`, `test_dataloader(videos_per_gpu=1)`
- Optimizer: `SGD(lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)`
- LR schedule: `CosineAnnealing` (`by_epoch=False`, `min_lr=0`)
- Epochs: `total_epochs=20 for full finetuning on large AI generated dataset; 10 for finetuning on real-world dataset if the dataset is small`
- Evaluation: every epoch with `metrics=['f1']`, save best by `mean_f1`


<!-- You can use [vis_skeleton](/demo/vis_skeleton.ipynb) to visualize the provided skeleton data. -->

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
Examples,
```shell
bash tools/dist_train.sh configs/stgcn++/stgcn++_dtc_v2_yolo11/j_mult_label.py 1 --validate --test-best
bash tools/dist_test.sh configs/stgcn++/stgcn++_dtc_v2_yolo11/j_mult_label.py \
  work_dirs/stgcn++/stgcn++_dtc_multi-label-knk1/j_cl100_ml1_t5.0_ovlp0.0_seed0/best_mean_f1_epoch_*.pth \
  1 --out output/dtc_stgcnpp_eval.pkl --eval f1
```

## Finetune YOLO
### Data Processing
Convert DTC annotation style files into YOLO data format:
```
python tools/data/convert_pose_format.py
```
This will create a data folder with the following file structure
```
dataset_root/
 ├── images/
 │    ├── train/
 │    └── val/
 ├── labels/
 │    ├── train/
 │    └── val/
``` 
The user need to create a data.yaml file under dataset_root. The yaml template can be found at [dataset guide](https://docs.ultralytics.com/datasets/pose#supported-dataset-formats)

### Finetune
```
python tools/finetune_yolo.py
```
