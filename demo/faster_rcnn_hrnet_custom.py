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
from pyskl.utils.misc import load_video_info, analyze_video_data, print_analysis

from ultralytics import YOLO
import cv2

from pyskl.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'Failed to import `inference_detector` and `init_detector` from `mmdet.apis`. '
        'Make sure you can successfully import these if you want to use related features. '
    )

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
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
FONTSCALE = 1
FONTCOLOR = (0, 0, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('--video', help='video file/url', default='data/tmp/openart-video_23b522ff_1755939662590.mp4')
    parser.add_argument('--out_filename', help='output filename', default='output/openart1_cl48_2-person.mp4')
    parser.add_argument('--out_dir', default='output/hrnet-2person')
    parser.add_argument(
        '--config',
        default='configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_1x_coco-person.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
                 'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        # default='https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/label_map/nturgbd_120.txt',
        help='label map file')
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


def frame_extraction(video_path, short_side):
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

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1], poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    if num_joints is None:
        return None, None
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3), dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


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


def action_recognition_inference(args, config, model, video_info):
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [op for op in config.data.test.pipeline if op['type'] == 'FormatGCNInput'][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    out_dir = osp.join(args.out_dir, video_info['frame_dir'].split('/')[-2])
    os.makedirs(out_dir, exist_ok=True)
    out_filename = osp.join(out_dir, video_info['frame_dir'].split('/')[-1])

    frame_paths, original_frames = frame_extraction(video_info['frame_dir'],
                                                    args.short_side)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    # Get Human detection results
    det_results = detection_inference(args, frame_paths) # list[np.ndarray(num_person, 5)], len = num_frame
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results) # list[list[dict{'bbox':array(5), 'keypoints':array(17,3)}]], len = num_frame
    torch.cuda.empty_cache()

    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    if GCN_flag:
        # We will keep at most `GCN_nperson` persons per frame.
        tracking_inputs = [[pose['keypoints'] for pose in poses] for poses in pose_results]
        keypoint, keypoint_score = pose_tracking(tracking_inputs, max_tracks=GCN_nperson)
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score
    else:
        num_person = max([len(x) for x in pose_results])
        # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                            dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                  dtype=np.float16)
        for i, poses in enumerate(pose_results):
            for j, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[j, i] = pose[:, :2]
                keypoint_score[j, i] = pose[:, 2]
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

    if fake_anno['keypoint'] is None:
        action_label = ''
    else:
        results = inference_recognizer(model, fake_anno)
        action_label = label_map[results[0][0]]

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]

    # tracking_poses = []
    # for i in range(num_frame):
    #     # poses = [{"keypoints": np.concatenate((keypoint[p][i], keypoint_score[p][i].reshape(-1, 1)), axis=1)} for p in range(keypoint.shape[0])]
    #     poses = [{"keypoints": np.concatenate((keypoint[p][i], keypoint_score[p][i].reshape(-1, 1)), axis=1)} for p in range(2)]
    #     tracking_poses.append(poses)
    # vis_frames = [
    #     vis_pose_result(pose_model, frame_paths[i], tracking_poses[i])
    #     for i in range(num_frame)
    # ]

    for frame in vis_frames:
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=video_info['fps'])
    vid.write_videofile(out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


def main():
    args = parse_args()

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [x for x in config.data.test.pipeline if x['type'] != 'DecompressPose']

    model = init_recognizer(config, args.checkpoint, args.device)

    video_infos = load_video_info(dir_path="data/DTC/AI-videos-selective")
    # # analyze video infos
    # result = analyze_video_data(video_infos)
    # print_analysis(results=result)

    for video_info in video_infos:
        video_path = video_info['frame_dir'] # 'data/DTC/AI-videos-selective/jump-flap-lie/openart-video_d20266ad_17555.mp4'
        print(f'Processing video: {video_path}')
        action_recognition_inference(args, config, model, video_info)



if __name__ == '__main__':
    main()
