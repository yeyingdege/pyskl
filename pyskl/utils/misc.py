# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import hashlib
import logging
import multiprocessing as mp
import numpy as np
import os
import cv2
import os.path as osp
import socket
import warnings
import statistics
from collections import Counter
from mmcv import load
from mmcv.runner import get_dist_info
from mmcv.utils import get_logger

from ..smp import download_file


def mc_on(port=22077, launcher='pytorch', size=60000, min_size=6):
    # size is mb, allocate 24GB memory by default.
    mc_exe = 'memcached' if launcher == 'pytorch' else '/mnt/lustre/share/memcached/bin/memcached'
    os.system(f'{mc_exe} -p {port} -m {size}m -I {min_size}m -d')


def cache_file(arg_tuple):
    mc_cfg, data_file = arg_tuple
    assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
    retry = 3
    while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
        time.sleep(5)
        retry -= 1
    assert retry >= 0, 'Failed to launch memcached. '
    from pymemcache import serde
    from pymemcache.client.base import Client

    cli = Client(mc_cfg, serde=serde.pickle_serde)

    if isinstance(data_file, str):
        assert osp.exists(data_file)
        kv_dict = load(data_file)
    else:
        if not isinstance(data_file, dict):
            assert isinstance(data_file[0], tuple) and len(data_file[0]) == 2
            data_file = {k: v for k, v in data_file}
        kv_dict = data_file

    if isinstance(kv_dict, list):
        assert ('frame_dir' in kv_dict[0]) != ('filename' in kv_dict[0])
        key = 'frame_dir' if 'frame_dir' in kv_dict[0] else 'filename'
        kv_dict = {x[key]: x for x in kv_dict}
    for k, v in kv_dict.items():
        flag = None
        while not isinstance(flag, dict):
            try:
                cli.set(k, v)
            except:
                cli = Client(mc_cfg, serde=serde.pickle_serde)
                cli.set(k, v)
            try:
                flag = cli.get(k)
            except:
                cli = Client(mc_cfg, serde=serde.pickle_serde)
                flag = cli.get(k)


def mp_cache(mc_cfg, mc_list, num_proc=32):
    args = [(mc_cfg, x) for x in mc_list]
    pool = mp.Pool(num_proc)
    pool.map(cache_file, args)


def mp_cache_single(mc_cfg, file_name, num_proc=32):
    data = load(file_name)
    assert 'annotations' in data
    annos = data['annotations']
    tups = [(x['frame_dir'], x) for x in annos]
    tups = [tups[i::num_proc] for i in range(num_proc)]
    args = [(mc_cfg, tup_list) for tup_list in tups]
    pool = mp.Pool(num_proc)
    pool.map(cache_file, args)


def mc_off():
    os.system('killall memcached')


def test_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    assert isinstance(ip, str)
    if isinstance(port, str):
        port = int(port)
    assert 1 <= port <= 65535
    result = sock.connect_ex((ip, port))
    return result == 0


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "pyskl".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


def cache_checkpoint(filename, cache_dir='.cache'):
    if filename.startswith('http://') or filename.startswith('https://'):
        url = filename.split('//')[1]
        basename = filename.split('/')[-1]
        filehash = hashlib.md5(url.encode('utf8')).hexdigest()[-8:]
        os.makedirs(cache_dir, exist_ok=True)
        local_pth = osp.join(cache_dir, basename.replace('.pth', f'_{filehash}.pth'))
        if not osp.exists(local_pth):
            download_file(filename, local_pth)
        filename = local_pth
    return filename

def warning_r0(warn_str):
    rank, _ = get_dist_info()
    if rank == 0:
        warnings.warn(warn_str)


def get_video_info_opencv(video_path):
    """
    Get video information using OpenCV
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': frame_count,
        'width': width,
        'height': height,
        'duration_seconds': duration
    }


def load_video_info(dir_path):
    """
    Load video file information:
        "frame_dir": filename or full path, 
        "total_frames", "fps", 
        "original_shape": (height, width),
        "coarse_labels"
    """
    video_info = []
    for subdir in os.listdir(dir_path):
        # subdir ='jump-flap-lie'
        coarse_labels = subdir.split('-')
        subdir_path = os.path.join(dir_path, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    # read video
                    info = get_video_info_opencv(os.path.join(subdir_path, filename))
                    d = {
                        'filename': filename,
                        'frame_dir': os.path.join(subdir_path, filename),
                        'coarse_labels': coarse_labels
                    }
                    d |= info
                    video_info.append(d)
    return video_info


# Calculate statistics
def get_stats(data_list, name):
    if not data_list or all(x == 0 for x in data_list):
        return {f"{name}_count": 0}
    
    return {
        f"{name}_count": len(data_list),
        f"{name}_min": min(data_list),
        f"{name}_max": max(data_list),
        f"{name}_mean": round(statistics.mean(data_list), 2),
        f"{name}_median": round(statistics.median(data_list), 2),
        f"{name}_stdev": round(statistics.stdev(data_list), 2) if len(data_list) > 1 else 0,
        f"{name}_sum": sum(data_list)
    }

def analyze_video_data(video_list):
    """
    Analyze a list of video dictionaries and return comprehensive statistics.
    
    Args:
        video_list (list): List of dictionaries containing video metadata
        
    Returns:
        dict: Dictionary containing various statistics about the video data
    """
    
    if not video_list:
        return {"error": "No data provided"}
    
    # Initialize collectors
    total_frames_list = []
    fps_list = []
    duration_list = []
    width_list = []
    height_list = []
    all_labels = []
    label_combinations = []
    filenames = []
    frame_dirs = []
    
    # Collect data from each video
    for video in video_list:
        total_frames_list.append(video.get('total_frames', 0))
        fps_list.append(video.get('fps', 0))
        duration_list.append(video.get('duration_seconds', 0))
        width_list.append(video.get('width', 0))
        height_list.append(video.get('height', 0))
        
        # Handle labels
        labels = video.get('coarse_labels', [])
        all_labels.extend(labels)
        label_combinations.append(tuple(sorted(labels)))  # Sort for consistent comparison
        
        filenames.append(video.get('filename', ''))
        frame_dirs.append(video.get('frame_dir', ''))

    # Compile results
    results = {
        "dataset_overview": {
            "total_videos": len(video_list),
            "unique_filenames": len(set(filenames)),
            "unique_frame_dirs": len(set(frame_dirs))
        },
        
        "total_frames_stats": get_stats(total_frames_list, "total_frames"),
        
        "fps_stats": get_stats(fps_list, "fps"),
        
        "duration_stats": get_stats(duration_list, "duration"),
        
        "resolution_stats": {
            **get_stats(width_list, "width"),
            **get_stats(height_list, "height"),
            "unique_resolutions": list(set(zip(width_list, height_list))),
            "resolution_counts": dict(Counter(zip(width_list, height_list)))
        },
        
        "label_analysis": {
            "total_label_instances": len(all_labels),
            "unique_labels": list(set(all_labels)),
            "label_frequency": dict(Counter(all_labels)),
            "unique_label_combinations": len(set(label_combinations)),
            "label_combination_frequency": dict(Counter(label_combinations)),
            "videos_per_label": {
                label: sum(1 for video in video_list if label in video.get('coarse_labels', []))
                for label in set(all_labels)
            }
        }
    }
    
    return results

def print_analysis(results):
    """Pretty print the analysis results"""
    
    print("=" * 60)
    print("VIDEO DATASET ANALYSIS")
    print("=" * 60)
    
    # Dataset Overview
    print("\n📊 DATASET OVERVIEW:")
    overview = results["dataset_overview"]
    print(f"  Total Videos: {overview['total_videos']}")
    print(f"  Unique Filenames: {overview['unique_filenames']}")
    print(f"  Unique Frame Directories: {overview['unique_frame_dirs']}")
    
    # Total Frames Statistics
    print("\n🎬 TOTAL FRAMES STATISTICS:")
    frames = results["total_frames_stats"]
    print(f"  Count: {frames['total_frames_count']}")
    print(f"  Range: {frames['total_frames_min']} - {frames['total_frames_max']}")
    print(f"  Mean: {frames['total_frames_mean']}")
    print(f"  Median: {frames['total_frames_median']}")
    print(f"  Std Dev: {frames['total_frames_stdev']}")
    print(f"  Total: {frames['total_frames_sum']:,} frames")
    
    # FPS Statistics
    print("\n🎥 FPS STATISTICS:")
    fps = results["fps_stats"]
    print(f"  Range: {fps['fps_min']} - {fps['fps_max']}")
    print(f"  Mean: {fps['fps_mean']}")
    print(f"  Median: {fps['fps_median']}")
    
    # Duration Statistics
    print("\n⏱️  DURATION STATISTICS:")
    duration = results["duration_stats"]
    print(f"  Range: {duration['duration_min']:.2f}s - {duration['duration_max']:.2f}s")
    print(f"  Mean: {duration['duration_mean']:.2f}s")
    print(f"  Median: {duration['duration_median']:.2f}s")
    print(f"  Total Duration: {duration['duration_sum']:.2f}s ({duration['duration_sum']/60:.1f} minutes)")
    
    # Resolution Statistics
    print("\n📺 RESOLUTION STATISTICS:")
    res = results["resolution_stats"]
    print(f"  Width Range: {res['width_min']} - {res['width_max']}")
    print(f"  Height Range: {res['height_min']} - {res['height_max']}")
    print(f"  Unique Resolutions: {res['unique_resolutions']}")
    print("  Resolution Distribution:")
    for resolution, count in res['resolution_counts'].items():
        print(f"    {resolution[0]}x{resolution[1]}: {count} videos")
    
    # Label Analysis
    print("\n🏷️  LABEL ANALYSIS:")
    labels = results["label_analysis"]
    print(f"  Total Label Instances: {labels['total_label_instances']}")
    print(f"  Unique Labels ({len(labels['unique_labels'])}): {sorted(labels['unique_labels'])}")
    # print("  Label Frequency:")
    # for label, freq in sorted(labels['label_frequency'].items(), key=lambda x: x[1], reverse=True):
    #     print(f"    '{label}': {freq} occurrences")
    
    # print(f"\n  Unique Label Combinations: {labels['unique_label_combinations']}")
    # print("  Label Combination Frequency:")
    # for combo, freq in labels['label_combination_frequency'].items():
    #     combo_str = ", ".join(combo) if combo else "No labels"
    #     print(f"    [{combo_str}]: {freq} videos")
    
    print("\n  Videos Per Label:")
    for label, count in sorted(labels['videos_per_label'].items(), key=lambda x: x[1], reverse=True):
        print(f"    '{label}': appears in {count} videos")
