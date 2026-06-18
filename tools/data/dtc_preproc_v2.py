import os
import sys
import json
import argparse
import random
import numpy as np
import mmcv
from collections import defaultdict
from tqdm import tqdm
from pyskl.utils.misc import load_video_info, analyze_video_data, print_analysis

sys.path.append(os.getcwd())
from tools.data.segment_action import get_video_fps_and_total_frame, segment_actions_by_duration, segment_actions_by_duration_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--video_dir', type=str, default='data/DTC/AI-videos-selective-Sep30')
    parser.add_argument('--multi_label', default=True, help='multi-label classification')
    parser.add_argument('--out_dir', default='data/DTC/multi-label-knk1', help='multi-label skeleton dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed for data split')
    parser.add_argument('--keep_labels', nargs='+', type=str,
                        default=['fall', 'hit', 'kick', 'run', 'throw'])
    parser.add_argument('--label_map', type=str, default='tools/data/label_map/dtc7.txt')
    parser.add_argument('--annotation_dir', type=str, default='data/DTC/annotations_1.0_knk_1',
                        help='directory of annotation json files')
    parser.add_argument('--frame_thres', type=int, default=30, help='minimum frames for a valid video segment')
    parser.add_argument('--min_action_frames', type=int, default=15, help='minimum action overlap frames for multi-label segmentation')
    parser.add_argument('--segment_duration', type=float, default=5.0, help='duration (in seconds)')
    parser.add_argument('--overlap_duration', type=float, default=0.0, help='overlap duration (in seconds)')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='fraction of files assigned to train (default 0.8)')
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Stratified split helpers
# ---------------------------------------------------------------------------

def get_file_labels(file_path):
    """Return the set of unique action labels present in one annotation file."""
    try:
        with open(file_path, 'r') as f:
            ann = json.load(f)
    except Exception:
        return set()
    labels = set()
    for person in ann.get('annotation', []):
        for action_ann in person.get('action_annotation', []):
            for lbl in action_ann.get('label', []):
                labels.add(lbl)
    return labels


def stratified_split(files, annotation_dir, train_ratio=0.6, seed=10):
    """
    Assign each file to train or val so that every action label appears in
    both splits at approximately *train_ratio* : *(1-train_ratio)*.

    Strategy
    --------
    1. Build a mapping  label -> [files containing that label].
    2. Sort labels by frequency (rarest first) so rare classes are handled
       with care.
    3. For each label (rarest-first), if the current train fraction for that
       label is below the target, assign the next un-assigned file that
       carries it to train; otherwise to val.
    4. Any file still unassigned after the label pass is split randomly to
       hit the global ratio.
    """
    rng = random.Random(seed)

    # --- build label -> file list ------------------------------------------
    file_labels: dict[str, set] = {}
    label_files: dict[str, list] = defaultdict(list)

    for fname in files:
        fpath = os.path.join(annotation_dir, fname)
        lbls = get_file_labels(fpath)
        file_labels[fname] = lbls
        for lbl in lbls:
            label_files[lbl].append(fname)

    # shuffle each label's file list for randomness
    for lbl in label_files:
        rng.shuffle(label_files[lbl])

    train_set: set[str] = set()
    val_set: set[str] = set()

    # sort labels rarest → most common
    sorted_labels = sorted(label_files.keys(), key=lambda l: len(label_files[l]))

    for lbl in sorted_labels:
        lbl_files = label_files[lbl]
        # count how many of this label's files are already assigned
        already_train = sum(1 for f in lbl_files if f in train_set)
        already_val   = sum(1 for f in lbl_files if f in val_set)
        unassigned    = [f for f in lbl_files if f not in train_set and f not in val_set]

        for f in unassigned:
            total_so_far = already_train + already_val + 1
            desired_train = round(total_so_far * train_ratio)
            if already_train < desired_train:
                train_set.add(f)
                already_train += 1
            else:
                val_set.add(f)
                already_val += 1

    # files with no recognised labels — split by global ratio
    unassigned_rest = [f for f in files if f not in train_set and f not in val_set]
    rng.shuffle(unassigned_rest)
    cut = round(len(unassigned_rest) * train_ratio)
    train_set.update(unassigned_rest[:cut])
    val_set.update(unassigned_rest[cut:])

    # preserve a stable order for reproducibility
    train_files = [f for f in files if f in train_set]
    val_files   = [f for f in files if f in val_set]

    return train_files, val_files


def report_split_distribution(files, annotation_dir, split_name):
    """Print per-label file counts for a given file list."""
    label_counts: dict[str, int] = defaultdict(int)
    for fname in files:
        fpath = os.path.join(annotation_dir, fname)
        for lbl in get_file_labels(fpath):
            label_counts[lbl] += 1
    print(f"\n  [{split_name}] {len(files)} files — label distribution (files per label):")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: x[1]):
        print(f"    {lbl}: {cnt}")
    return label_counts


# ---------------------------------------------------------------------------
# Annotation readers
# ---------------------------------------------------------------------------

def read_single_annotation(file_path, label_to_id, frame_thres=10, multi_label=False):
    with open(file_path, 'r') as f:
        ann = json.load(f)
    frame_dir = ann.get('frame_dir', '')
    frame_dir = frame_dir.replace('/data/dtc/dataset/Sep30/AI-videos-selective', 'data/DTC/AI-videos-selective-Sep30')
    frame_dir = frame_dir.replace('/data/dtc/dataset/', 'data/DTC/')
    filename = frame_dir.split('/')[-1]
    annotation = ann.get('annotation', [])
    new_anns = []
    ids = []
    for person in annotation:
        person_id = person.get('person_id', -1)
        bbox = person.get('bbox', [])
        keypoints = person.get('keypoint', [])
        keypoint_score = person.get('keypoint_score', [])
        action_anns = person['action_annotation']
        if isinstance(keypoints, list) and len(keypoints) > 0:
            keypoints = np.array(keypoints, dtype=np.float16)
        if isinstance(keypoint_score, list) and len(keypoint_score) > 0:
            keypoint_score = np.array(keypoint_score, dtype=np.float16)
        for i, action_ann in enumerate(action_anns):
            label_text = action_ann['label']
            start_frame = action_ann['start_frame']
            end_frame = action_ann['end_frame']
            assert len(label_text) == 1, "Only single-label supported"
            emulated_filename = f"{filename[:-4]}_pid{person_id}_act{i}_{start_frame}_{end_frame}"
            ann_dict = {
                'filename': emulated_filename,
                'frame_dir': frame_dir,
                'person_id': person_id,
                'img_shape': (ann['original_shape'][0], ann['original_shape'][1]),
                'original_shape': (ann['original_shape'][0], ann['original_shape'][1]),
                'modality': 'Pose',
                'keypoint': np.expand_dims(keypoints[start_frame:end_frame+1, :, :], axis=0), # (1, T, K, 3)
                'keypoint_score': np.expand_dims(keypoint_score[start_frame:end_frame+1, :], axis=0), # (1, T, K),
                'bbox': np.array(bbox, dtype=np.float16)
            }
            ann_dict['total_frames'] = ann_dict['keypoint'].shape[1]
            ann_dict['label'] = [label_to_id.get(label_text[0], 0)] if multi_label else label_to_id.get(label_text[0], 0)
            if ann_dict['total_frames'] < frame_thres:
                continue
            new_anns.append(ann_dict)
            ids.append(emulated_filename)
    return new_anns, ids


def read_single_annotation_multi_label(file_path, label_to_id, segment_duration,
                                       overlap_duration, frame_thres, min_action_frames,
                                       is_train=True):
    with open(file_path, 'r') as f:
        ann = json.load(f)
    frame_dir = ann.get('frame_dir', '')
    # change directory to local directory
    frame_dir = frame_dir.replace('/data/dtc/dataset/Sep30/AI-videos-selective', 'data/DTC/AI-videos-selective-Sep30')
    frame_dir = frame_dir.replace('/data/dtc/dataset/', 'data/DTC/')
    filename = frame_dir.split('/')[-1]
    annotation = ann.get('annotation', [])
    new_anns = []
    ids = []
    for person in annotation:
        person_id = person.get('person_id', -1)
        bbox = person.get('bbox', [])
        keypoints = person.get('keypoint', [])
        keypoint_score = person.get('keypoint_score', [])
        action_anns = person['action_annotation']
        if isinstance(keypoints, list) and len(keypoints) > 0:
            keypoints = np.array(keypoints, dtype=np.float16)
        if isinstance(keypoint_score, list) and len(keypoint_score) > 0:
            keypoint_score = np.array(keypoint_score, dtype=np.float16)
        if keypoints.shape[0] < frame_thres:
            continue
        try:
            fps, total_frames = get_video_fps_and_total_frame(frame_dir)
        except Exception:
            total_frames = ann.get('total_frames', keypoints.shape[0])
            fps = 20
        if is_train: # Action-aligned windows: Segments start at each action's start_frame
            segments = segment_actions_by_duration_v2(
                action_anns, fps, total_frames,
                segment_duration=segment_duration,
                overlap=overlap_duration,
                min_segment_frames=frame_thres,
                min_overlap_frames=min_action_frames # minimum action overlap of 15 frames
            )
        else: # fixed window and stride segmentation
            segments = segment_actions_by_duration(
                action_anns, fps, total_frames,
                segment_duration=segment_duration,
                overlap=overlap_duration,
                min_segment_frames=frame_thres,
                min_overlap_frames=min_action_frames
            )
        for i, seg in enumerate(segments):
            start_frame = seg['start_frame']
            end_frame = seg['end_frame']
            labels_in_segment = seg['labels']
            labels = set()
            for label_text in labels_in_segment:
                for lt in label_text['label']:
                    label = label_to_id.get(lt, 0)
                    if label != 0:
                        labels.add(label)
            labels = list(labels) if labels else [0]
            emulated_filename = f"{filename[:-4]}_pid{person_id}_seg{i}_{start_frame}_{end_frame}"
            seg_keypoints = keypoints[start_frame:end_frame, :, :]
            seg_keypoint_score = keypoint_score[start_frame:end_frame, :]
            seg_bbox = bbox[start_frame:end_frame]
            if seg_keypoints.shape[0] < frame_thres:
                continue
            ann_dict = {
                'filename': emulated_filename,
                'frame_dir': frame_dir,
                'person_id': person_id,
                'img_shape': (ann['original_shape'][0], ann['original_shape'][1]),
                'original_shape': (ann['original_shape'][0], ann['original_shape'][1]),
                'modality': 'Pose',
                'label': labels,
                'keypoint': np.expand_dims(seg_keypoints, axis=0), # (1, T, K, 3)
                'keypoint_score': np.expand_dims(seg_keypoint_score, axis=0), # (1, T, K)
                'total_frames': seg_keypoints.shape[0],
                'bbox': np.array(seg_bbox, dtype=np.float16)
            }
            new_anns.append(ann_dict)
            ids.append(emulated_filename)
    return new_anns, ids


def read_annotations(dir_path, label_to_id, files=None, multi_label=True,
                     segment_duration=5.0, overlap_duration=0.0,
                     frame_thres=30, min_action_frames=15, is_train=True):
    """Read a directory that has all json annotation files."""
    anns = []
    ids = []
    if files is None:
        files = os.listdir(dir_path)
    for file in files:
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(dir_path, file)
        if multi_label:
            ann, id = read_single_annotation_multi_label(
                file_path, label_to_id, segment_duration, overlap_duration,
                frame_thres, min_action_frames, is_train,
            )
        else:
            ann, id = read_single_annotation(
                file_path, label_to_id, frame_thres, multi_label,
            )
        anns += ann
        ids += id
    return anns, ids


def main():
    args = parse_args()

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    label_to_id = {label: i for i, label in enumerate(label_map)}
    print('Label map:', label_to_id)

    random.seed(args.seed)

    # --- stratified split ---------------------------------------------------
    all_files = [f for f in os.listdir(args.annotation_dir) if f.endswith('.json')]
    # Shuffle first so ties within the greedy pass are resolved randomly
    random.shuffle(all_files)

    train_files, val_files = stratified_split(
        all_files, args.annotation_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(f'\nVideos total {len(all_files)}, '
          f'train {len(train_files)}, val {len(val_files)} '
          f'(ratio {len(train_files)/len(all_files):.2f}:{len(val_files)/len(all_files):.2f})')

    # report per-label distribution in each split
    train_label_counts = report_split_distribution(train_files, args.annotation_dir, 'train')
    val_label_counts   = report_split_distribution(val_files,   args.annotation_dir, 'val')

    print('\n  Achieved train/(train+val) ratio per label:')
    all_labels = sorted(set(train_label_counts) | set(val_label_counts))
    for lbl in all_labels:
        tr = train_label_counts.get(lbl, 0)
        vl = val_label_counts.get(lbl, 0)
        ratio = tr / (tr + vl) if (tr + vl) > 0 else float('nan')
        print(f'    {lbl}: train={tr}, val={vl}, train_ratio={ratio:.2f}')

    # --- build annotation dicts --------------------------------------------
    train_anns, train_ids = read_annotations(
        args.annotation_dir, label_to_id,
        files=train_files, multi_label=args.multi_label,
        segment_duration=args.segment_duration, overlap_duration=args.overlap_duration,
        frame_thres=args.frame_thres, min_action_frames=args.min_action_frames,
        is_train=True,
    )
    val_anns, val_ids = read_annotations(
        args.annotation_dir, label_to_id,
        files=val_files, multi_label=args.multi_label,
        segment_duration=args.segment_duration, overlap_duration=args.overlap_duration,
        frame_thres=args.frame_thres, min_action_frames=args.min_action_frames,
        is_train=False,
    )

    result = analyze_video_data(train_anns + val_anns)
    print_analysis(results=result)

    print(f'\nSegments total {len(train_ids) + len(val_ids)}, '
          f'train {len(train_anns)}, val {len(val_anns)}')

    data = {
        'split': {'train': train_ids, 'val': val_ids},
        'annotations': train_anns + val_anns,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(
        args.out_dir,
        f"dtc{len(label_map)}_ml{int(args.multi_label)}"
        f"_t{args.segment_duration}_ovlp{args.overlap_duration}"
        f"_seed{args.seed}_stratified.pkl",
    )
    mmcv.dump(data, out_file)
    print('Saved file', out_file)

    # result = analyze_video_data(train_anns)
    # print_analysis(results=result)
    # result = analyze_video_data(val_anns)
    # print_analysis(results=result)


if __name__ == '__main__':
    main()
