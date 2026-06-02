# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import csv
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
import subprocess

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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo (long-video sliding-window)')
    parser.add_argument('--video', help='video file / annotation .json/.pkl',
                        default='output_segments/output_027.mp4')
    parser.add_argument('--out_dir', help='output directory', default='output/')
    parser.add_argument(
        '--config',
        default='configs/stgcn++/stgcn++_dtc_v2_yolo11/j_mult_label_demo.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default='work_dirs/stgcn++/stgcn++_dtc_multi-label-seg3/j_cl100_ml1_t5.0_ovlp0.0_seed0/best_mean_f1_epoch_13.pth',
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
        default='https://download.openmmlab.com/mmpose/top_down/hrnet/'
                'hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side', type=int, default=480,
        help='specify the short-side length of the image')
    # ---- sliding-window parameters ----
    parser.add_argument(
        '--window-size', type=float, default=5.0,
        help='duration of each inference window in seconds (default: 5.0)')
    parser.add_argument(
        '--window-stride', type=float, default=5.0,
        help='stride between consecutive windows in seconds (default: 5.0, '
             'i.e. non-overlapping). Use a value < window-size for overlap.')
    parser.add_argument(
        '--fps', type=float, default=25,
        help='video FPS. Auto-detected for .mp4/.avi/.mov; required for '
             'annotation files (.json/.pkl) when no video is available.')
    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_video(frames, out_filename, fps=24):
    if isinstance(frames, str):
        filenames = os.listdir(frames)
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        frames = [cv2.imread(osp.join(frames, f)) for f in filenames]
    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in frames], fps=fps)
    vid.write_videofile(out_filename, remove_temp=True)
    return vid


def frame_extraction(video_path, short_side, resize=True):
    """Extract every frame from *video_path* into ./tmp/<video_stem>/."""
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames, frame_paths = [], []
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


def get_video_fps(video_path):
    """Return the FPS reported by the video file header."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 24.0


def format_timestamp(seconds):
    """Convert a float number of seconds to a human-readable HH:MM:SS.mmm string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


# ---------------------------------------------------------------------------
# Pose extraction (YOLO, full video at once)
# ---------------------------------------------------------------------------

def extract_pose(yolo_model, video_path):
    """Run YOLO-pose tracking on the full video and return packed arrays."""
    results = yolo_model.track(source=video_path, persist=False,
                               conf=0.5, iou=0.7, device='0', classes=[0], verbose=False)
    pose_results = []
    person_ids = set()
    total_boxes = 0
    for r in results:
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            pose_results.append([])
            continue
        pose_result = []
        for id_, box, kpts in zip(r.boxes.id.cpu().numpy(),
                                  r.boxes.xyxy.cpu().numpy(),
                                  r.keypoints.data.cpu().numpy()):
            pid = int(id_) if id_ is not None else -1
            pose_result.append({'id': pid, 'bbox': box, 'keypoints': kpts})
            person_ids.add(pid)
            total_boxes += 1
        pose_results.append(pose_result)

    person_id_to_idx = {pid: idx for idx, pid in enumerate(sorted(person_ids))}
    idx_to_person_id = {idx: pid for pid, idx in person_id_to_idx.items()}
    num_frame = len(results)
    num_person = len(person_ids)
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for pose in poses:
            j = person_id_to_idx[pose['id']]
            keypoint[j, i] = pose['keypoints'][:, :2]
            try:
                keypoint_score[j, i] = pose['keypoints'][:, 2]
            except IndexError:
                print("Using 2d keypoint estimation without confidence scores; defaulting to 1.0")
                keypoint_score[j, i] = 1.0
    print(f"Extracted total {total_boxes} boxes. Average {total_boxes / num_frame:.4f} boxes/frame")
    return keypoint, keypoint_score, pose_results, idx_to_person_id


# ---------------------------------------------------------------------------
# Sliding-window helpers
# ---------------------------------------------------------------------------

def build_windows(num_frame, fps, window_size, window_stride):
    """
    Return a list of (start_frame, end_frame) tuples (end exclusive) covering
    the full sequence with the requested window size and stride.

    The last window is always extended/clamped to reach the final frame so no
    frames are dropped, even if the video length is not a multiple of the stride.
    """
    win_frames = max(1, int(round(window_size * fps)))
    stride_frames = max(1, int(round(window_stride * fps)))
    windows = []
    start = 0
    while start < num_frame:
        end = min(start + win_frames, num_frame)
        windows.append((start, end))
        if end == num_frame:
            break
        start += stride_frames
    return windows


def slice_keypoints(keypoint, keypoint_score, start_frame, end_frame):
    """Slice the per-person keypoint arrays to a frame range."""
    return (
        keypoint[:, start_frame:end_frame, :, :],
        keypoint_score[:, start_frame:end_frame, :],
    )


def run_window_inference(model, base_anno, kp_win, kp_score_win, label_map):
    """
    Build a fake_anno for a single window and run the recogniser.

    Returns
    -------
    person_actions : dict  {person_id -> list[str]}  e.g. {0: ['walk(0.91)']}
    raw_scores     : list[list[float]]  parallel to person order
    """
    num_person_win, num_frame_win = kp_win.shape[:2]
    if num_person_win == 0 or num_frame_win == 0:
        return {}, []

    fake_anno = dict(
        frame_dir=base_anno['frame_dir'],
        label=-1,
        img_shape=base_anno['img_shape'],
        original_shape=base_anno['original_shape'],
        start_index=0,
        modality='Pose',
        total_frames=num_frame_win,
        keypoint=kp_win,
        keypoint_score=kp_score_win,
        test_mode=True,
    )
    results, scores = inference_recognizer_parallel(model, fake_anno)
    action_labels = [[label_map[a] for a in res] for res in results]

    person_actions = {}
    raw_scores = []
    for idx, (al, sc) in enumerate(zip(action_labels, scores)):
        pid = base_anno['idx_to_person_id'][idx]
        person_actions[pid] = [f'{a}({s:.2f})' for a, s in zip(al, sc) if s >= 0.5]
        # if len(person_actions[pid]) == 0:
        #     person_actions[pid] = ['none']
        raw_scores.append(sc.tolist() if hasattr(sc, 'tolist') else list(sc))

    return person_actions, raw_scores


# ---------------------------------------------------------------------------
# CSV summary writer
# ---------------------------------------------------------------------------

def save_predictions_csv(all_window_results, label_map, out_path):
    """
    Write a CSV with one row per (window, person) containing:
        window_start, window_end, person_id, <label_0_score>, <label_1_score>, ...
    """
    fieldnames = ['window_start', 'window_end', 'person_id'] + label_map
    os.makedirs(osp.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_window_results:
            writer.writerow(entry)
    print(f'Predictions saved to {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [
        x for x in config.data.test.pipeline if x['type'] != 'DecompressPose'
    ]

    model = init_recognizer(config, args.checkpoint, args.device)
    # yolo_model = YOLO('.cache/yolo11m-pose.pt')
    # yolo_model = YOLO('.cache/yolo26l-pose.pt')
    # yolo_model = YOLO("runs/pose/train7/weights/best.pt") # custom-trained on DTC 2d keypoints
    yolo_model = YOLO("runs/pose/train2/weights/best.pt") # custom-trained on DTC 3d keypoints for 10 epochs
    # yolo_model = YOLO("runs/pose/train/weights/best.pt") # custom-trained on DTC 3d keypoints for 50 epochs

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    label_to_id = {label: i for i, label in enumerate(label_map)}

    # ------------------------------------------------------------------
    # 1. Load / extract ALL pose data for the full input
    # ------------------------------------------------------------------
    if args.video.endswith('.json') or args.video.endswith('.pkl'):
        anno_list, _ = read_single_annotation_multi_label(args.video, label_to_id)
        filename = anno_list[0]['frame_dir']

        # Try to load accompanying video frames
        frame_paths, original_frames = None, None
        if os.path.exists(filename) and filename.endswith(('.mp4', '.avi', '.mov')):
            frame_paths, original_frames = frame_extraction(filename, args.short_side, resize=False)
            fps = args.fps or get_video_fps(filename)
        else:
            fps = args.fps
            if fps is None:
                raise ValueError(
                    'Cannot determine FPS from annotation file alone. '
                    'Please supply --fps <value>.'
                )

        h, w = anno_list[0]['original_shape']
        num_person = len(anno_list)
        num_frame = anno_list[0]['total_frames']
        num_keypoint = 17
        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2), dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint), dtype=np.float16)
        idx_to_person_id = {}
        for i, ann in enumerate(anno_list):
            pid = ann['person_id']
            kpt = ann['keypoint']         # (1, T, K, 2)
            kpt_s = ann['keypoint_score']  # (1, T, K)
            keypoint[i] = kpt[0, :, :, :2]
            keypoint_score[i] = kpt_s[0]
            idx_to_person_id[i] = pid

        pose_results = []
        for fid in range(num_frame):
            frame_poses = []
            for pid_idx in range(num_person):
                frame_poses.append({
                    'id': idx_to_person_id[pid_idx],
                    'bbox': anno_list[pid_idx]['bbox'][fid],
                    'keypoints': np.hstack((
                        keypoint[pid_idx, fid],
                        keypoint_score[pid_idx, fid, :, np.newaxis],
                    )),
                })
            pose_results.append(frame_poses)

    elif args.video.endswith(('.mp4', '.avi', '.mov')):
        frame_paths, original_frames = frame_extraction(args.video, args.short_side, resize=False)
        filename = args.video
        num_frame = len(frame_paths)
        h, w, _ = original_frames[0].shape
        fps = args.fps or get_video_fps(filename)

        keypoint, keypoint_score, pose_results, idx_to_person_id = extract_pose(
            yolo_model, args.video
        )
        torch.cuda.empty_cache()
    else:
        raise ValueError(f'Unsupported input format: {args.video}')

    print(f'Total frames: {num_frame}  |  FPS: {fps:.2f}  |  '
          f'Duration: {num_frame / fps:.1f} s')

    # ------------------------------------------------------------------
    # 2. Build sliding windows and run inference per window
    # ------------------------------------------------------------------
    base_anno = dict(
        frame_dir=filename,
        img_shape=(h, w),
        original_shape=(h, w),
        idx_to_person_id=idx_to_person_id,
    )

    windows = build_windows(num_frame, fps, args.window_size, args.window_stride)
    print(f'Running inference over {len(windows)} window(s) '
          f'(size={args.window_size}s, stride={args.window_stride}s)')

    # frame_to_person_actions[frame_idx] = person_actions dict for that frame
    frame_to_person_actions = {}
    all_window_results = []   # for CSV export

    for win_idx, (start_f, end_f) in enumerate(windows):
        t_start = start_f / fps
        t_end = end_f / fps
        ts_start = format_timestamp(t_start)
        ts_end = format_timestamp(t_end)
        print(f'\n[Window {win_idx + 1}/{len(windows)}] '
              f'frames {start_f}–{end_f - 1}  ({ts_start} → {ts_end})')

        kp_win, kp_score_win = slice_keypoints(keypoint, keypoint_score, start_f, end_f)

        person_actions, raw_scores = run_window_inference(
            model, base_anno, kp_win, kp_score_win, label_map
        )

        for pid, actions in person_actions.items():
            print(f'  Person {pid}: {actions}')

        # Map this window's predictions to every frame in the window
        for fid in range(start_f, end_f):
            # Later windows overwrite earlier ones in the overlap region,
            # giving priority to the most recent window (straightforward for
            # non-overlapping; for overlapping consider averaging instead).
            frame_to_person_actions[fid] = person_actions

        # Collect results for CSV
        for idx, (pid, sc) in enumerate(
            zip(sorted(idx_to_person_id.values()), raw_scores)
        ):
            row = {
                'window_start': ts_start,
                'window_end': ts_end,
                'person_id': pid,
            }
            # Store per-label scores; pad / trim to match label_map length
            for lbl, s in zip(label_map, sc):
                row[lbl] = round(s, 4)
            all_window_results.append(row)

    # ------------------------------------------------------------------
    # 3. Save CSV summary
    # ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = osp.join(args.out_dir,
                        f"{osp.splitext(osp.basename(filename))[0]}_predictions.csv")
    save_predictions_csv(all_window_results, label_map, csv_path)

    # ------------------------------------------------------------------
    # 4. Visualise: render annotated video if frames are available
    # ------------------------------------------------------------------
    if frame_paths is None:
        print('No video frames available for visualisation — skipping.')
        return

    try:
        pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
        vis_frames = []
        for frame_idx, (frame_path, pose_result) in enumerate(
            zip(frame_paths, pose_results)
        ):
            t_sec = frame_idx / fps
            frame = cv2.imread(frame_path)
            annotated = frame.copy()
            annotated = vis_pose_result(pose_model, annotated, pose_result,
                                        radius=6, thickness=2)

            # Timestamp overlay (top-left)
            cv2.putText(annotated, format_timestamp(t_sec),
                        (30, 60), FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)

            # Action labels for each person in this frame's window
            person_actions_for_frame = frame_to_person_actions.get(frame_idx, {})
            annotated = visualize_action_each_person(
                annotated, pose_result, person_actions_for_frame
            )
            vis_frames.append(annotated)

        cv2.destroyAllWindows()

        out_filename = osp.join(
            args.out_dir,
            f"demo_{osp.splitext(osp.basename(filename))[0]}.mp4"
        )
        write_video(vis_frames, out_filename=out_filename, fps=fps)
        print(f'Annotated video saved to {out_filename}')

    except Exception as e:
        print(f'Pose visualisation failed: {e}')

    # ------------------------------------------------------------------
    # 5. Clean up temporary frame directory
    # ------------------------------------------------------------------
    tmp_frame_dir = osp.join('./tmp', osp.basename(osp.splitext(filename)[0]))
    if osp.isdir(tmp_frame_dir):
        shutil.rmtree(tmp_frame_dir)



def split_video(input_file, segment_length_seconds, output_dir="output_segments"):
    """
    Splits a video into non-overlapping segments of a specific length.
    """
    # Create an output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # FFmpeg command
    # -i: input file
    # -f segment: use the segment muxer
    # -segment_time: length of each piece in seconds
    # -reset_timestamps 1: make each segment start at 00:00
    # -c copy: copy streams without re-encoding (ultra fast)
    try:
        command = [
            'ffmpeg',
            '-i', input_file,
            '-f', 'segment',
            '-segment_time', str(segment_length_seconds),
            '-reset_timestamps', '1',
            '-c', 'copy',
            f'{output_dir}/output_%03d.mp4'
        ]
    except Exception as e:
        print(f"Error preparing FFmpeg command: {e}")
        command = [
            'ffmpeg',
            '-i', input_file,
            '-f', 'segment',
            '-segment_time', str(segment_length_seconds),
            '-reset_timestamps', '1',
            '-c:v', 'copy',      # Copy the video (fast)
            '-c:a', 'aac',       # Convert the incompatible audio to AAC
            f'{output_dir}/output_%03d.mp4'
        ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully split '{input_file}' into segments.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()
    # Usage: Split into 10-minute (600 seconds) chunks
    # split_video("data/DTC/school/D03_20260409094037.mp4", 300)