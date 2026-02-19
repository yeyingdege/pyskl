import os
import sys
import json
import argparse
import random
import numpy as np
import mmcv
from tqdm import tqdm
from pyskl.utils.yolo_utils import visualize_results
from pyskl.utils.misc import load_video_info, analyze_video_data, print_analysis

sys.path.append(os.getcwd())
from tools.data.segment_action import get_video_fps_and_total_frame, segment_actions_by_duration, segment_actions_by_duration_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--video_dir', type=str, default='data/DTC/AI-videos-selective-Sep30')
    # parser.add_argument('--multi_label', action='store_true', help='multi-label classification')
    parser.add_argument('--multi_label', default=True, help='multi-label classification')
    parser.add_argument('--out_dir', default='data/DTC/multi-label-seg', help='multi-label skeleton dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed for data split')
    parser.add_argument('--keep_labels', nargs='+', type=str, 
                        default=['fall', 'hit', 'kick', 'run', 'throw'])
    parser.add_argument('--label_map', type=str, default='tools/data/label_map/dtc7.txt')
    parser.add_argument('--annotation_dir', type=str, default='data/DTC/annotations_1.0_1764299882239', help='directory of annotation json files')
    parser.add_argument('--frame_thres', type=int, default=30, help='minimum frames for a valid video segment')
    parser.add_argument('--min_action_frames', type=int, default=15, help='minimum action overlap frames for multi-label segmentation')
    parser.add_argument('--segment_duration', type=float, default=5.0, help='duration (in seconds)')
    parser.add_argument('--overlap_duration', type=float, default=0.0, help='overlap duration (in seconds)')
    args = parser.parse_args()
    return args


def read_single_annotation(file_path, label_to_id, frame_thres=10, multi_label=False):
    with open(file_path, 'r') as f:
        ann = json.load(f)
    frame_dir = ann.get('frame_dir', '')
    # change directory to local directory
    frame_dir = frame_dir.replace('/data/dtc/dataset/Sep30/AI-videos-selective', 'data/DTC/AI-videos-selective-Sep30')
    filename = frame_dir.split('/')[-1]
    annotation = ann.get('annotation', [])
    new_anns = []
    ids = []
    for person in annotation:
        person_id = person.get('person_id', -1)
        bbox = person.get('bbox', []) # list(list)
        keypoints = person.get('keypoint', []) # list(list(list))
        keypoint_score = person.get('keypoint_score', []) # list(list)
        action_anns = person['action_annotation'] # list(dict)
        # convert list to np array
        if isinstance(keypoints, list) and len(keypoints) > 0:
            keypoints = np.array(keypoints, dtype=np.float16)
        if isinstance(keypoint_score, list) and len(keypoint_score) > 0:
            keypoint_score = np.array(keypoint_score, dtype=np.float16)
        # every action segment as a training sample
        for i, action_ann in enumerate(action_anns):
            label_text = action_ann['label'] # list
            start_frame = action_ann['start_frame']
            end_frame = action_ann['end_frame']
            # truncate keypoints, keypoint_score, bbox
            assert len(label_text) == 1, "Only single-label supported"
            emulated_filename = f"{filename[:-4]}_pid{person_id}_act{i}_{start_frame}_{end_frame}"
            ann_dict = {
                'filename': emulated_filename, # emulate different filenames for multiple labels from same video
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
            # assert ann_dict['keypoint'].shape[1] == total_frames, "Keypoint length mismatch"
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
    filename = frame_dir.split('/')[-1]
    annotation = ann.get('annotation', [])
    new_anns = []
    ids = []
    for person in annotation:
        person_id = person.get('person_id', -1)
        bbox = person.get('bbox', []) # list(list)
        keypoints = person.get('keypoint', []) # list(list(list))
        keypoint_score = person.get('keypoint_score', []) # list(list)
        action_anns = person['action_annotation'] # list(dict)
        # convert list to np array
        if isinstance(keypoints, list) and len(keypoints) > 0:
            keypoints = np.array(keypoints, dtype=np.float16)
        if isinstance(keypoint_score, list) and len(keypoint_score) > 0:
            keypoint_score = np.array(keypoint_score, dtype=np.float16)
        
        if  keypoints.shape[0] < frame_thres:
            continue

        # get fps and total frames
        fps, total_frames = get_video_fps_and_total_frame(frame_dir)
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
            labels_in_segment = seg['labels'] # list of labels in this segment
            labels = set()
            for label_text in labels_in_segment:
                lts = label_text['label']
                for lt in lts:
                    label = label_to_id.get(lt, 0)
                    if label == 0:
                        continue
                    labels.add(label)
            labels = list(labels)
            if len(labels) == 0:
                labels = [0]

            emulated_filename = f"{filename[:-4]}_pid{person_id}_seg{i}_{start_frame}_{end_frame}"
            seg_keypoints = keypoints[start_frame:end_frame, :, :]
            seg_keypoint_score = keypoint_score[start_frame:end_frame, :]
            seg_bbox = bbox[start_frame:end_frame]
            if seg_keypoints.shape[0] < frame_thres:
                continue
            ann_dict = {
                'filename': emulated_filename, # emulate different filenames for each person from same video
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
            ann, id = read_single_annotation_multi_label(file_path, label_to_id, segment_duration, overlap_duration, 
                                                         frame_thres, min_action_frames, is_train)
        else:
            ann, id = read_single_annotation(file_path, label_to_id, frame_thres, multi_label)
        anns = anns + ann
        ids = ids + id
    return anns, ids


def main():
    args = parse_args()

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    label_to_id = {label: i for i, label in enumerate(label_map)}
    print(label_to_id)

    # set random seed
    random.seed(args.seed)
    # split videos into train/val
    files = os.listdir(args.annotation_dir)
    random.shuffle(files)
    total = len(files)
    train_num = int(0.8 * total)
    train = files[:train_num]
    val = files[train_num:]
    print(f'Videos total {total}, train {len(train)}, val {len(val)}')

    train_anns, train_ids = read_annotations(args.annotation_dir, 
                                             label_to_id, 
                                             files=train, 
                                             multi_label=args.multi_label,
                                             segment_duration=args.segment_duration,
                                             overlap_duration=args.overlap_duration,
                                             frame_thres=args.frame_thres, 
                                             min_action_frames=args.min_action_frames,
                                             is_train=True)
    val_anns, val_ids = read_annotations(args.annotation_dir, 
                                         label_to_id, 
                                         files=val, 
                                         multi_label=args.multi_label,
                                         segment_duration=args.segment_duration,
                                         overlap_duration=args.overlap_duration,
                                         frame_thres=args.frame_thres, 
                                         min_action_frames=args.min_action_frames,
                                         is_train=False)
    
    # analyze video infos
    result = analyze_video_data(train_anns + val_anns)
    print_analysis(results=result)

    ids = train_ids + val_ids
    print(f'total {len(ids)}, train {len(train_anns)}, val {len(val_anns)}')
    data = {
        'split': {'train': train_ids, 'val': val_ids},
        'annotations': train_anns + val_anns
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"dtc{len(label_map)}_ml{int(args.multi_label)}_t{args.segment_duration}_ovlp{args.overlap_duration}_seed{args.seed}.pkl")
    mmcv.dump(data, out_file)
    print('Saved file', out_file)

    # result = analyze_video_data(train_anns)
    # print_analysis(results=result)
    # result = analyze_video_data(val_anns)
    # print_analysis(results=result)


if __name__ == '__main__':
    main()
