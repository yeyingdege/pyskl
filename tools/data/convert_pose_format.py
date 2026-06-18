import os
import json
import cv2
import random
import argparse
import pickle
import decord
import numpy as np
from PIL import Image
from pyskl.utils.yolo_utils import draw_skeleton


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract keypoints dataset from videos and annotations')
    parser.add_argument('--video_dir', type=str, default='data/DTC/AI-videos-selective-Sep30')
    parser.add_argument('--out_dir', default='data/gen-yolo-pose3d-0.4', help='keypoints extracted by yolo')
    parser.add_argument('--seed', type=int, default=0, help='random seed for data split')
    parser.add_argument('--label_map', type=str, default='tools/data/label_map/dtc7.txt')
    parser.add_argument('--annotation_dir', type=str, default='data/DTC/annotations_upto_feb19', help='directory of annotation json files')
    args = parser.parse_args()
    return args


coco2017_val_ann_path = './data/person_keypoints_val2017.json'
'''
dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
images 0/5000:
{'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
annotations 0/11004:
segmentation: <class 'list'>, length 1
num_keypoints: <class 'int'>
area: <class 'float'>
iscrowd: <class 'int'>
keypoints: <class 'list'>, length 51
image_id: <class 'int'>
bbox: <class 'list'>, length 4, [x, y, w, h]
category_id: <class 'int'>
id: <class 'int'>
Unique image ids: 5000, Unique annotation ids: 11004
Unique image ids in annotations: 2693
'''

def print_coco_format(ann_path):
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)

    print(coco_data.keys())
    # images format
    img = coco_data['images'][0]
    print(f"images 0/{len(coco_data['images'])}:\n{img}")
    # annotations format
    ann = coco_data['annotations'][0]
    print(f"annotations 0/{len(coco_data['annotations'])}:")
    for k, v in ann.items():
        s = f"{k}: {type(v)}"
        if type(v) == list:
            s += f", length {len(v)}"
        print(s)

    image_ids = set(img['id'] for img in coco_data['images'])
    ann_ids = set(ann['id'] for ann in coco_data['annotations'])
    ann_image_ids = set(ann['image_id'] for ann in coco_data['annotations'])
    print(f"Unique image ids: {len(image_ids)}, Unique annotation ids: {len(ann_ids)}")
    print(f"Unique image ids in annotations: {len(ann_image_ids)}")


def convert_coco_bbox_to_xyxy(bbox):
    x1, y1 = int(np.floor(bbox[0])), int(np.floor(bbox[1]))
    x2, y2 = int(np.ceil(bbox[0] + bbox[2])), int(np.ceil(bbox[1] + bbox[3]))
    return [x1, y1, x2, y2]

def convert_coco_keypoints(keypoints):
    # coco keypoints format: [x1, y1, v1, x2, y2, v2, ...]
    # where v is visibility (0=not labeled, 1=labeled but not visible, 2=labeled and visible)
    kpts_xy = [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 3)]
    kpts_conf = [keypoints[i+2] for i in range(0, len(keypoints), 3)]
    return kpts_xy, kpts_conf


def visualize_bbox_and_keypoints(ann, img_path):
    # coco_url: "http://images.cocodataset.org/val2017/000000425226.jpg"
    x1, y1, x2, y2 = convert_coco_bbox_to_xyxy(ann['bbox'])
    frame = cv2.imread(img_path)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
    kpts_xy, kpts_conf = convert_coco_keypoints(ann['keypoints'])
    frame = draw_skeleton(frame, kpts_xy, kpts_conf, keypoint_thr=0)
    cv2.imwrite("bbox_keypoints.jpg", frame)
    # cv2.imshow("bbox", frame)
    # cv2.waitKey(0)
    return


def select_videos(anno_dir):
    # exclude videos without key action labels.
    files = os.listdir(anno_dir)
    files_selected = []
    for file in files:
        if not file.endswith('.json'):
            continue
        file_path = os.path.join(anno_dir, file)
        with open(file_path, 'r') as f:
            ann = json.load(f)
        annotation = ann.get('annotation', [])
        video_actions = []
        for person in annotation:
            action_anns = person['action_annotation'] # list(dict)
            for person_action in action_anns:
                video_actions += person_action['label']
        video_actions = list(set(video_actions))
        if len(video_actions) == 1 and video_actions[0] == 'others':
            continue
        files_selected.append(file)
    print(f"Total files before filtering: {len(files)}. Files after filtering: {len(files_selected)}")
    return files_selected


def process_timestamp_overlapping(timestamps):
    if not timestamps:
        return []

    # Sort timestamps based on the start time
    sorted_timestamps = sorted(timestamps, key=lambda x: x[0])
    
    merged_segments = [list(sorted_timestamps[0])]
    
    for start, end in sorted_timestamps[1:]:
        last_merged = merged_segments[-1]
        
        # If the current segment overlaps with the last merged segment, merge them
        if start <= last_merged[1]:
            last_merged[1] = max(last_merged[1], end)
        else:
            merged_segments.append([start, end])
    return merged_segments


def select_frames(file_path, total_sample_frames=20):
    with open(file_path, 'r') as f:
        ann = json.load(f)
    start_end_frames = []
    annotation = ann.get('annotation', [])
    for person in annotation:
        action_anns = person['action_annotation']
        action_anns = sorted(action_anns, key=lambda x: x['start_frame'])
        for person_action in action_anns:
            start_frame = person_action['start_frame']
            end_frame = person_action['end_frame']
            start_end_frames.append([start_frame, end_frame])

    start_end_frames = process_timestamp_overlapping(start_end_frames)

    frame_keypoints_and_bbox = []
    for i in range(ann['total_frames']):
        frame_info = {'keypoints': [], 'keypoint_score': [], 'bbox': []}
        for person in annotation:
            frame_info['keypoints'].append(person['keypoint'][i])
            frame_info['keypoint_score'].append(person['keypoint_score'][i])
            frame_info['bbox'].append(person['bbox'][i])
        frame_keypoints_and_bbox.append(frame_info)

    def flatten(x):
        """Recursively flatten nested lists into a single list of values."""
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    def count_active_people(frame_info):
        """Count people whose bbox AND keypoints are both non-zero."""
        count = 0
        for bbox, kps in zip(frame_info['bbox'], frame_info['keypoints']):
            bbox_active = any(v != 0 for v in flatten(bbox))
            kps_active  = any(v != 0 for v in flatten(kps))
            if bbox_active and kps_active:
                count += 1
        return count

    # Collect all unique frame indices that fall within the merged action periods
    total_frames = ann['total_frames']
    action_frame_set = set()
    for start, end in start_end_frames:
        # action_frame_set.update(range(start, end + 1))
        clamped_start = max(0, min(start, total_frames - 1))
        clamped_end   = max(0, min(end,   total_frames - 1))
        action_frame_set.update(range(clamped_start, clamped_end + 1))
    action_frames = sorted(action_frame_set)

    if not action_frames:
        return []

    # If fewer candidate frames than requested, return them all
    if len(action_frames) <= total_sample_frames:
        print("Selected all frames without sampling.")
        return action_frames

    # Divide candidate frames into `total_sample_frames` equal buckets.
    # From each bucket pick the frame that has the most active people,
    # ensuring fixed-interval temporal coverage while preferring quality frames.
    n = total_sample_frames
    bucket_size = len(action_frames) / n
    selected = []
    for i in range(n):
        bucket_start = int(i * bucket_size)
        bucket_end   = int((i + 1) * bucket_size)
        bucket = action_frames[bucket_start:bucket_end]
        if not bucket:
            continue
        best = max(bucket, key=lambda idx: count_active_people(frame_keypoints_and_bbox[idx]))
        selected.append(best)
    selected = sorted(set(selected))
    keypoints_and_bboxes = [frame_keypoints_and_bbox[idx] for idx in selected]
    return selected, keypoints_and_bboxes


def select_frames_split(files, file_dir):
    frame_dict = {} # key: video_id, value: frame indexs
    for file in files:
        file_path = os.path.join(file_dir, file)
        video_frames, keypoints_and_bboxes = select_frames(file_path)
        frame_dict[file] = {
            'frame_indices': video_frames,
            'keypoints_and_bboxes': keypoints_and_bboxes
        }
    return frame_dict


def count_frames(data):
    total = 0
    for v in data.values():
        total += len(v['frame_indices'])
    return total


def read_frames_given_idxs(video_path, frame_idxs):
    """
    frame_idxs: frame number in raw video
    """
    reader = decord.VideoReader(video_path)
    vlen1 = len(reader)
    frame_idxs[-1] = min(frame_idxs[-1], vlen1)
    frame_idxs = np.array(frame_idxs)

    frames = reader.get_batch(frame_idxs).asnumpy()
    return frames


def extract_frames(info, image_dir, ann_dir):
    """Extract frames and return image info in coco format
    """
    curr_image_id = 0 # dict
    curr_kpt_id = 0
    coco_image_infos = []
    coco_annotation_infos = []
    for filename, file_info in info.items():
        frame_indices = file_info['frame_indices'] # list
        keypoints_and_bboxes = file_info['keypoints_and_bboxes']
        ann = json.load(open(os.path.join(ann_dir, filename), 'r'))
        video_path = ann['frame_dir']
        video_path = video_path.replace('/data/dtc/dataset/Sep30/AI-videos-selective', 'data/DTC/AI-videos-selective-Sep30')
        video_path = video_path.replace('/data/dtc/dataset/', 'data/DTC/')
        video_name = video_path.split('/')[-1].split('.')[0] # openart-video_0a3e2b79_1755655479860
        width, height = ann['original_shape'][1], ann['original_shape'][0]
        frames = read_frames_given_idxs(video_path, frame_indices)
        for frame, kpts_and_box in zip(frames, keypoints_and_bboxes):
            has_keypoints = False
            # convert keyponts and bbox to coco format. filter kpts using keypoint score.
            for kpts, kpt_score, bbox in zip(kpts_and_box['keypoints'], kpts_and_box['keypoint_score'], kpts_and_box['bbox']):
                coco_kpts = convert_to_coco_format(kpts, kpt_score)
                if all(v == 0 for v in coco_kpts):
                    continue
                coco_annotation_infos.append({
                    "id": curr_kpt_id,
                    "image_id": curr_image_id,
                    "category_id": 1, # person
                    "keypoints": coco_kpts,
                    "num_keypoints": sum(1 for score in kpt_score if score >= 0.8),
                    "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    "iscrowd": 0
                })
                has_keypoints = True
                curr_kpt_id += 1
            if has_keypoints:
                image_name = f"{video_name}_{curr_image_id:06d}.jpg"
                coco_image_infos.append({
                    "id": curr_image_id,
                    "file_name": image_name,
                    "width": width,
                    "height": height,
                    "license": 0
                })
                image_path = os.path.join(image_dir, image_name)
                img = Image.fromarray(frame)
                img.save(image_path)
                curr_image_id += 1
    return coco_image_infos, coco_annotation_infos


def convert_to_coco_format(kpts, kpt_score, kpt_score_thr=0.8):
    # kpts: list of 17 keypoints, each keypoint is [x, y]
    # kpt_score: list of 17 keypoint scores
    coco_kpts = []
    for i in range(len(kpts)):
        if kpt_score[i] < kpt_score_thr:
            coco_kpts.extend([0, 0, 0]) # not labeled
        else:
            coco_kpts.extend([kpts[i][0], kpts[i][1], 2]) # labeled and visible
    return coco_kpts


def convert_to_yolo_format(kpts, kpt_score, bbox, width, height, kpt_score_thr=0.4):
    row = [0]
    yolo_kpts = []
    for i in range(len(kpts)):
        if kpt_score[i] < kpt_score_thr:
            yolo_kpts.extend([kpts[i][0] / width, kpts[i][1] / height, 0]) # not labeled
        else:
            # yolo_kpts.extend([kpts[i][0] / width, kpts[i][1] / height])
            yolo_kpts.extend([kpts[i][0] / width, kpts[i][1] / height, kpt_score[i]])
    x_center = (bbox[0] + bbox[2]) / 2 / width
    y_center = (bbox[1] + bbox[3]) / 2 / height
    bbox_width = (bbox[2] - bbox[0]) / width
    bbox_height = (bbox[3] - bbox[1]) / height
    row.extend([x_center, y_center, bbox_width, bbox_height])
    row.extend(yolo_kpts)
    row_str = ' '.join(str(x) for x in row)
    return row_str, yolo_kpts


def extract_frames_yolo_format(info, image_dir, ann_dir, dtc_ann_dir):
    """Extract frames and save annotations in yolo format
    """
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    curr_image_id = 0 # dict
    total_rows = 0
    for filename, file_info in info.items():
        frame_indices = file_info['frame_indices'] # list
        keypoints_and_bboxes = file_info['keypoints_and_bboxes']
        ann = json.load(open(os.path.join(dtc_ann_dir, filename), 'r'))
        video_path = ann['frame_dir']
        video_path = video_path.replace('/data/dtc/dataset/Sep30/AI-videos-selective', 'data/DTC/AI-videos-selective-Sep30')
        video_path = video_path.replace('/data/dtc/dataset/', 'data/DTC/')
        video_name = video_path.split('/')[-1].split('.')[0] # openart-video_0a3e2b79_1755655479860
        width, height = ann['original_shape'][1], ann['original_shape'][0]
        frames = read_frames_given_idxs(video_path, frame_indices)
        for frame, kpts_and_box in zip(frames, keypoints_and_bboxes):
            has_keypoints = False
            yolo_ann_rows = []
            # convert keyponts and bbox to yolo format. filter kpts using keypoint score.
            for kpts, kpt_score, bbox in zip(kpts_and_box['keypoints'], kpts_and_box['keypoint_score'], kpts_and_box['bbox']):
                yolo_row, yolo_kpts = convert_to_yolo_format(kpts, kpt_score, bbox, width, height)
                if all(v == 0 for v in yolo_kpts):
                    continue
                yolo_ann_rows.append(yolo_row)
                has_keypoints = True
            if has_keypoints:
                image_name = f"{video_name}_{curr_image_id:08d}.jpg"
                image_path = os.path.join(image_dir, image_name)
                img = Image.fromarray(frame)
                img.save(image_path)
                # write annotation file
                ann_name = f"{video_name}_{curr_image_id:08d}.txt"
                ann_path = os.path.join(ann_dir, ann_name)
                with open(ann_path, 'w') as f:
                    f.write('\n'.join(yolo_ann_rows))
                total_rows += len(yolo_ann_rows)
                curr_image_id += 1
    print(f"Total images processed: {curr_image_id + 1}")
    print(f"Total rows written: {total_rows}")
    return 


def main():
    args = parse_args()

    label_map = [x.strip() for x in open(args.label_map).readlines()]
    label_to_id = {label: i for i, label in enumerate(label_map)}
    print(label_to_id)

    out_file = os.path.join(args.out_dir, 'sampled_frame_info.pkl')
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(out_file):
        # files_selected = select_videos(args.annotation_dir) # Total files before filtering: 392. Files after filtering: 391
        files = os.listdir(args.annotation_dir)
        random.seed(args.seed)
        random.shuffle(files)

        total = len(files)
        train_num = int(0.8 * total)
        train_files = files[:train_num]
        val_files = files[train_num:]
        # select frames
        frame_dict_train = select_frames_split(train_files, args.annotation_dir) # 313
        frame_dict_val = select_frames_split(val_files, args.annotation_dir) # 79
        print(f"Total number of frames from training set: {count_frames(frame_dict_train)}") # 6260
        print(f"Total number of frames from val set: {count_frames(frame_dict_val)}") # 1580
        # save info into a file
        info = {'train': frame_dict_train, 'val': frame_dict_val}
        with open(out_file, 'wb') as f:
            pickle.dump(info, f)
        print(f"Saved file to {out_file}")
    else:
        info = pickle.load(open(out_file, 'rb'))

    # extract and save images
    train_image_dir = os.path.join(args.out_dir, 'images', 'train')
    val_image_dir = os.path.join(args.out_dir, 'images', 'val')
    train_ann_dir = os.path.join(args.out_dir, 'labels', 'train')
    val_ann_dir = os.path.join(args.out_dir, 'labels', 'val')

    extract_frames_yolo_format(info['train'], train_image_dir, train_ann_dir, args.annotation_dir)
    extract_frames_yolo_format(info['val'], val_image_dir, val_ann_dir, args.annotation_dir)

    # coco_image_infos_train, coco_annotation_infos_train = extract_frames(info['train'], os.path.join(image_dir, 'train'), args.annotation_dir)
    # coco_image_infos_val, coco_annotation_infos_val = extract_frames(info['val'], os.path.join(image_dir, 'val'), args.annotation_dir)
    # # save coco format json
    # coco_train = {
    #     "info": {},
    #     "licenses": [],
    #     "images": coco_image_infos_train,
    #     "annotations": coco_annotation_infos_train,
    #     "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
    # }
    # coco_val = {
    #     "info": {},
    #     "licenses": [],
    #     "images": coco_image_infos_val,
    #     "annotations": coco_annotation_infos_val,
    #     "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
    # }
    # with open(os.path.join(args.out_dir, 'coco_format_train.json'), 'w') as f:
    #     json.dump(coco_train, f)
    # with open(os.path.join(args.out_dir, 'coco_format_val.json'), 'w') as f:
    #     json.dump(coco_val, f)
    # print(f"Saved coco format json to {args.out_dir}")


if __name__ == '__main__':
    # print_coco_format(coco2017_val_ann_path)
    # with open(coco2017_val_ann_path, 'r') as f:
    #     coco_data = json.load(f)
    # ann = coco_data['annotations'][0]
    # visualize_bbox_and_keypoints(ann, "data/000000425226.jpg")
    main()