import os
import cv2
from ultralytics import YOLO
from demo.my_demo import write_video


# COCO keypoint skeleton (pairs of keypoint indices)
# Index order: [0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
#               5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
#               9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
#               13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle]
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # arms
    (11, 12), (5, 11), (6, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
    (0, 1), (0, 2), (1, 3), (2, 4)              # head
]

LEFT_KEYPOINTS = [1, 3, 5, 7, 9, 11, 13, 15]   # left side indices
RIGHT_KEYPOINTS = [2, 4, 6, 8, 10, 12, 14, 16]  # right side indices
CENTER_KEYPOINTS = [0]  # nose

# Color scheme
LEFT_COLOR = (0, 255, 255)    # Cyan for left side
RIGHT_COLOR = (255, 0, 255)   # Magenta for right side
CENTER_COLOR = (0, 255, 0)    # Green for center/nose

# Skeleton line colors
LEFT_SKELETON_COLOR = (0, 200, 200)
RIGHT_SKELETON_COLOR = (200, 0, 200)
CENTER_SKELETON_COLOR = (150, 200, 0)


def run_demo(yolo_path=".cache/yolo11m-pose.pt", video="demo/ntu_sample.avi"):
    # Load pretrained pose model
    model = YOLO(yolo_path)  # or yolo11s-pose, yolo11l-pose etc.
    # results = model.track(source=video, stream=True, persist=True) # stream mode
    results = model.track(source=video, persist=True)
    return results


def visualize_results(results, out_file="yolo11m-pose_demo.mp4", action_label='', fps=24, keypoint_thr=0.1):
    vis_frames = []
    for frame_id, result in enumerate(results):
        frame = result.orig_img.copy()  # original frame
        boxes = result.boxes  # bounding boxes + track IDs
        keypoints = result.keypoints  # pose keypoints

        if boxes is not None and keypoints is not None:
            for box, kpts in zip(boxes, keypoints):
                # Track ID
                track_id = int(box.id.item()) if box.id is not None else -1
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
                # Person ID label
                label = f"ID_{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
                # Keypoints
                kpts_xy = kpts.xy[0].tolist()   # [(x,y), ...]
                kpts_conf = kpts.conf[0].tolist()  # confidence for each keypoint

                # # Draw keypoints
                # for (x, y), conf in zip(kpts_xy, kpts_conf):
                #     if conf > keypoint_thr:  # only draw confident keypoints
                #         cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 0), -1)

                # # Draw skeleton connections
                # for i, j in SKELETON:
                #     if kpts_conf[i] > keypoint_thr and kpts_conf[j] > keypoint_thr:
                #         pt1 = (int(kpts_xy[i][0]), int(kpts_xy[i][1]))
                #         pt2 = (int(kpts_xy[j][0]), int(kpts_xy[j][1]))
                #         cv2.line(frame, pt1, pt2, (255, 150, 0), 2)

                # Draw keypoints with different colors for left/right
                for idx, ((x, y), conf) in enumerate(zip(kpts_xy, kpts_conf)):
                    if conf > keypoint_thr:  # only draw confident keypoints
                        # Choose color based on keypoint position
                        if idx in LEFT_KEYPOINTS:
                            color = LEFT_COLOR
                        elif idx in RIGHT_KEYPOINTS:
                            color = RIGHT_COLOR
                        elif idx in CENTER_KEYPOINTS:
                            color = CENTER_COLOR
                        else:
                            color = (255, 255, 255)  # white for undefined
                        
                        cv2.circle(frame, (int(x), int(y)), 6, color, -1)

                # Draw skeleton connections with colored lines
                for i, j in SKELETON:
                    if kpts_conf[i] > keypoint_thr and kpts_conf[j] > keypoint_thr:
                        pt1 = (int(kpts_xy[i][0]), int(kpts_xy[i][1]))
                        pt2 = (int(kpts_xy[j][0]), int(kpts_xy[j][1]))
                        
                        # Determine line color based on connection type
                        if i in LEFT_KEYPOINTS and j in LEFT_KEYPOINTS:
                            line_color = LEFT_SKELETON_COLOR
                        elif i in RIGHT_KEYPOINTS and j in RIGHT_KEYPOINTS:
                            line_color = RIGHT_SKELETON_COLOR
                        else:
                            # Mixed or center connection
                            line_color = CENTER_SKELETON_COLOR
                        
                        cv2.line(frame, pt1, pt2, line_color, 2)

        if action_label != '':
            cv2.putText(frame, action_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        vis_frames.append(frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    write_video(vis_frames, out_filename=out_file, fps=fps)


def visualize_action_each_person(frame, pose_result, person_actions):
    for pose_dict in pose_result:
        pid = pose_dict['id']
        action_label = person_actions.get(pid, '')
        box = pose_dict['bbox']
        x1, y1, x2, y2 = map(int, box)

        # Person ID label
        label = f"ID{pid}_{action_label}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
    return frame



if __name__ == "__main__":
    video = "data/DTC/AI-videos-selective-Sep30/kick/openart-video_0121d561_1756984486262.mp4"
    results = run_demo(video=video)
    visualize_results(results, out_file="output/yolo11m-pose_openart-video_0121d561_1756984486262.mp4")
