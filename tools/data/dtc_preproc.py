import os
import json
import argparse
import random
import numpy as np
import mmcv
from tqdm import tqdm
from ultralytics import YOLO
from pyskl.utils.yolo_utils import visualize_results
from pyskl.utils.misc import load_video_info, analyze_video_data, print_analysis

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    parser.add_argument('--video_dir', type=str, default='data/DTC/AI-videos-selective-Sep30')
    parser.add_argument('--out', type=str, help='output pickle name',
                        default='data/DTC/dtc7.pkl')
    parser.add_argument('--keep_labels', nargs='+', type=str, 
                        default=['fall', 'hit', 'kick', 'run', 'throw'])
    parser.add_argument('--label_map', type=str, default='tools/data/label_map/dtc7.txt')
    parser.add_argument('--emulate', action='store_true', help='emulate label without loading annotations')
    parser.add_argument('--annotation_dir', type=str, default='data/DTC/annotations_1.0_1761210448056')
    parser.add_argument('--frame_thres', type=int, default=10, help='minimum frames for a valid action segment')
    args = parser.parse_args()
    return args


def read_single_annotation(file_path, label_to_id, frame_thres=10):
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
        # bbox = person.get('bbox', []) # list(list)
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
                'img_shape': (ann['original_shape'][0], ann['original_shape'][1]),
                'original_shape': (ann['original_shape'][0], ann['original_shape'][1]),
                'modality': 'Pose',
                'keypoint': np.expand_dims(keypoints[start_frame:end_frame+1, :, :], axis=0), # (1, T, K, 3)
                'keypoint_score': np.expand_dims(keypoint_score[start_frame:end_frame+1, :], axis=0), # (1, T, K)
                'label': label_to_id.get(label_text[0], 0)
            }
            ann_dict['total_frames'] = ann_dict['keypoint'].shape[1]
            if ann_dict['total_frames'] < frame_thres:
                continue
            # assert ann_dict['keypoint'].shape[1] == total_frames, "Keypoint length mismatch"
            new_anns.append(ann_dict)
            ids.append(emulated_filename)
    return new_anns, ids


def read_annotations(dir_path, label_to_id, frame_thres):
    """Read a directory that has all json annotation files."""
    anns = []
    ids = []
    for file in os.listdir(dir_path):
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(dir_path, file)
        ann, id = read_single_annotation(file_path, label_to_id, frame_thres)
        anns = anns + ann
        ids = ids + id
    return anns, ids


def extract_pose(yolo_model, video_info):
    results = yolo_model.track(source=video_info['frame_dir'], persist=False,
                               conf=0.5, iou=0.7, device='gpu', classes=[0], verbose=False)
    pose_results = []
    person_ids = set()
    for frame_id, r in enumerate(results):
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            continue
        pose_result = []
        for id, box, kpts in zip(r.boxes.id.cpu().numpy(),
                                 r.boxes.xyxy.cpu().numpy(), 
                                 r.keypoints.data.cpu().numpy()):
            pose_dict = {
                'id': int(id) if id is not None else -1,
                'bbox': box,
                'keypoints': kpts  # shape (num_keypoints, 3)
            }
            pose_result.append(pose_dict)
            person_ids.add(pose_dict['id'])
        pose_results.append(pose_result)

    # map discontinuous id to continuous indexes 0, 1, ...
    person_id_to_idx = {pid: idx for idx, pid in enumerate(sorted(person_ids))}
    num_frame = len(results)
    num_person = len(person_ids)
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)
    try:
        for i, poses in enumerate(pose_results): # for each frame
            for pose in poses:
                j = person_id_to_idx[pose['id']] # very important to pack keypoints correctly by person id
                keypoint[j, i, :, :] = pose['keypoints'][:, :2]
                keypoint_score[j, i, :] = pose['keypoints'][:, 2]
    except Exception as e:
        print(f'person id {person_ids}, num_person {num_person}')
        raise ValueError('Error in assigning keypoints.', e)

    video_info['keypoint'] = keypoint
    video_info['keypoint_score'] = keypoint_score
    video_info['img_shape'] = (video_info['height'], video_info['width'])
    video_info['original_shape'] = (video_info['height'], video_info['width'])
    video_info['modality'] = 'Pose'

    # visualize_results(results, out_file='tmp/tmp.mp4', 
    #                   action_label=','.join(video_info['coarse_labels']), fps=video_info['fps'],
    #                   keypoint_thr=0.1)
    return video_info


def main():
    args = parse_args()
    assert args.out.endswith('.pkl')

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    label_to_id = {label: i for i, label in enumerate(label_map)}
    print(label_to_id)

    if args.emulate:
        print('Emulating annotations without loading real annotations')

        video_infos = load_video_info(dir_path=args.video_dir)
        random.seed(0)
        random.shuffle(video_infos)
        yolo_model = YOLO(".cache/yolo11m-pose.pt")

        anns = []
        ids = []
        for video_info in tqdm(video_infos):
            ids.append(video_info['filename'])
            print("Processing video", video_info['filename'])
            video_info['label'] = [label_to_id[l] if l in label_to_id else 0 for l in video_info['coarse_labels']]
            ann = extract_pose(yolo_model, video_info)
            anns.append(ann)
    else:
        anns, ids = read_annotations(args.annotation_dir, label_to_id, args.frame_thres)
    
    # analyze video infos
    result = analyze_video_data(anns)
    print_analysis(results=result)

    total = len(ids)
    train_num = int(0.8 * total)
    train = ids[:train_num]
    val = ids[train_num:]
    print(f'total {total}, train {len(train)}, val {len(val)}')
    data = {
        'split': {'train': train, 'val': val},
        'annotations': anns
    }
    mmcv.dump(data, args.out)
    print('Saved file', args.out)


if __name__ == '__main__':
    main()
