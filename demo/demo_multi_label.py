# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
import warnings
from scipy.optimize import linear_sum_assignment
import sys
sys.path.append('.')

import cv2
from ultralytics import YOLO

from pyskl.apis import init_recognizer, inference_recognizer_parallel
from pyskl.utils.yolo_utils import visualize_action_each_person
from tools.data.dtc_preproc import read_single_annotation_multi_label

try:
    from mmpose.apis import init_pose_model, vis_pose_result
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'Failed to import `init_pose_model`, `inference_top_down_pose_model`, `vis_pose_result` from '
        '`mmpose.apis`. Make sure you can successfully import these if you want to use related features. '
    )

try:
    import moviepy as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 2
FONTCOLOR = (0, 0, 255)  # BGR, white
THICKNESS = 2
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('--video', help='video file/url', default='demo/ntu_sample.avi')
    parser.add_argument('--out_dir', help='output directory', default='output/')
    parser.add_argument(
        '--config',
        default='configs/stgcn++/stgcn++_dtc_v2_yolo11/j_mult_label_demo.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/stgcn++/stgcn++_dtc_v2/j_ml1_seed111/best_top1_acc_epoch_19.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/dtc7.txt',
        help='label map file')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    args = parser.parse_args()
    return args


def write_video(frames, out_filename, fps=24):
    if isinstance(frames, str):
        filenames = os.listdir(frames)
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        frames = [cv2.imread(osp.join(frames, f)) for f in filenames]
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in frames], fps=fps)
    vid.write_videofile(out_filename, remove_temp=True)
    return vid


def frame_extraction(video_path, short_side, resize=True):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
        if resize:
            frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def extract_pose(yolo_model, video_path):
    results = yolo_model.track(source=video_path, persist=False,
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
    idx_to_person_id = {idx: pid for pid, idx in person_id_to_idx.items()}
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

    return keypoint, keypoint_score, pose_results, idx_to_person_id


def play_video_with_pose(vis_frames, action_label, loop=10, fps=24):
    """Play video frames with pose and action label superimposed, looping N times."""
    delay = int(1000 / fps)
    for _ in range(loop):
        for frame in vis_frames:
            # Overlay action label (already done, but ensure it's visible)
            cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                        FONTCOLOR, THICKNESS, LINETYPE)
            cv2.imshow('Pose Estimation Playback', frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    model = init_recognizer(config, args.checkpoint, args.device)
    yolo_model = YOLO(".cache/yolo11m-pose.pt")
    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]
    label_to_id = {label: i for i, label in enumerate(label_map)}

    # handle skeleton input
    if args.video.endswith('.json') or args.video.endswith('.pkl'):
        anno_list, _ = read_single_annotation_multi_label(args.video, label_to_id)
        # merge multiple skeleton annotations into one during inference
        filename = anno_list[0]['frame_dir']
        if os.path.exists(filename) and filename.endswith(('.mp4', '.avi', '.mov')):
            frame_paths, original_frames = frame_extraction(filename, args.short_side, resize=False)
        h, w = anno_list[0]['original_shape']
        num_person = len(anno_list)
        num_frame = anno_list[0]['total_frames']
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)
        idx_to_person_id = {}
        for i, ann in enumerate(anno_list):
            person_id = ann['person_id']
            kpt = ann['keypoint'] # (1, T, K, 3)
            kpt_score = ann['keypoint_score'] # (1, T, K)
            keypoint[i, :, :, :] = kpt[0, :, :, :2]
            keypoint_score[i, :, :] = kpt_score[0, :, :]
            idx_to_person_id[i] = person_id

        pose_results = []
        for fid in range(num_frame):
            pose_result = []
            for pid in range(num_person):
                pose_dict = {
                    'id': idx_to_person_id[pid],
                    'bbox': anno_list[pid]['bbox'][fid],
                    'keypoints': np.hstack((keypoint[pid, fid, :, :], 
                                            keypoint_score[pid, fid, :][:, np.newaxis]))  # shape (num_keypoints, 3)
                }
                pose_result.append(pose_dict)
            pose_results.append(pose_result)
    
    elif args.video.endswith('.mp4') or args.video.endswith('.avi') or args.video.endswith('.mov'):
        # handle video input
        frame_paths, original_frames = frame_extraction(args.video, args.short_side, resize=False)
        filename = args.video
        num_frame = len(frame_paths)
        h, w, _ = original_frames[0].shape

        keypoint, keypoint_score, pose_results, idx_to_person_id = extract_pose(yolo_model, args.video)
        torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir=filename,
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame,
        keypoint=keypoint,
        keypoint_score=keypoint_score,
        test_mode=True)

    if fake_anno['keypoint'] is None:
        action_label = ''
    else:
        results, scores = inference_recognizer_parallel(model, fake_anno)
        # assume each person id has a prediction result
        action_label = [[label_map[a] for a in res] for res in results]
        person_actions = {}
        for pid, al, score in zip(sorted(idx_to_person_id.values()), action_label, scores):
            action_and_score_str = [f'{a}({s:.2f})' for a, s in zip(al, score)]
            person_actions[pid] = action_and_score_str
            print(f'Person ID {pid}: Predicted action: {action_and_score_str}')

    try:
        pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
        # draw keypoints and action label on each frame
        vis_frames = []
        frame_count = 0
        for frame_path, pose_result in zip(frame_paths, pose_results):
            frame = cv2.imread(frame_path)
            annotated = frame.copy()
            annotated = vis_pose_result(pose_model, annotated, pose_result,
                                                radius=6, thickness=2)
            cv2.putText(annotated, f"frame {frame_count+1}", (30, 60), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
            # draw each person's action
            annotated = visualize_action_each_person(annotated, pose_result, person_actions)
            vis_frames.append(annotated)
            frame_count += 1
        cv2.destroyAllWindows()

        os.makedirs(args.out_dir, exist_ok=True)
        out_filename = osp.join(args.out_dir, f"demo_{filename.split('/')[-1].split('.')[0]}.mp4")
        write_video(vis_frames, out_filename=out_filename, fps=24)
    except Exception as e:
        print('Pose visualization failed:', e)
    tmp_frame_dir = osp.join('./tmp', osp.basename(osp.splitext(filename)[0]))
    shutil.rmtree(tmp_frame_dir)



if __name__ == '__main__':
    main()

