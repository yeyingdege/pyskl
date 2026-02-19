"""
Function to segment variable-length action sequences into fixed-duration chunks.
Includes padding for short segments and minimum overlap thresholds.
"""
import cv2
from typing import List, Dict, Union


def get_video_fps_and_total_frame(video_path):
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
    cap.release()
    return fps, frame_count


def segment_actions_by_duration(
    action_labels: List[Dict[str, Union[str, int]]],
    fps: float,
    total_frames: int,
    segment_duration: float = 2.0,
    overlap: float = 0.0,
    min_segment_frames: int = 30,
    min_overlap_frames: int = 15
) -> List[Dict]:
    """
    Cut time segments of variant length into fixed-duration segments.
    
    Args:
        action_labels: List of dictionaries with keys:
            - 'label': str, action label name
            - 'start_frame': int, starting frame number
            - 'end_frame': int, ending frame number
        fps: float, frames per second of the video
        total_frames: int, total number of frames in the video
        segment_duration: float, desired segment duration in seconds (e.g., 2, 3, 4)
        overlap: float, overlap between consecutive segments in seconds (default: 0)
        min_segment_frames: int, minimum frames required for a segment (default: 20)
            - If remaining frames < min_segment_frames: discard
            - If remaining frames >= min_segment_frames: pad to segment_frames
        min_overlap_frames: int, minimum overlap frames to include an action label (default: 12)
    
    Returns:
        List of dictionaries with keys:
            - 'segment_id': int, sequential segment identifier
            - 'start_frame': int, starting frame of the segment
            - 'end_frame': int, ending frame of the segment
            - 'start_time': float, starting time in seconds
            - 'end_time': float, ending time in seconds
            - 'duration': float, actual duration in seconds
            - 'is_padded': bool, whether this segment was padded
            - 'labels': list of dicts containing actions in this segment with:
                - 'label': str, action label
                - 'original_start_frame': int, original action start frame
                - 'original_end_frame': int, original action end frame
                - 'overlap_start_frame': int, where action starts within segment
                - 'overlap_end_frame': int, where action ends within segment
                - 'overlap_frames': int, number of overlapping frames
                - 'coverage_ratio': float, what fraction of segment this action covers
    """
    if not action_labels:
        return []
    
    # Calculate segment length in frames
    segment_frames = int(segment_duration * fps)
    overlap_frames = int(overlap * fps)
    stride_frames = segment_frames - overlap_frames
    
    # # Find the overall start and end frames
    # min_frame = min(action['start_frame'] for action in action_labels)
    # max_frame = max(action['end_frame'] for action in action_labels)
    
    # # Use total_frames if provided, otherwise use max_frame
    # video_end_frame = min(total_frames, max_frame) if total_frames else max_frame
    
    segments = []
    segment_id = 0
    current_start_frame = 0
    video_end_frame = total_frames
    
    while current_start_frame < video_end_frame:
        current_end_frame = current_start_frame + segment_frames
        remaining_frames = video_end_frame - current_start_frame
        
        # Check if we should discard or pad this segment
        if remaining_frames < min_segment_frames:
            # Discard: not enough frames left
            break
        elif remaining_frames < segment_frames:
            # Pad: we have enough frames (>= min_segment_frames) but less than full segment
            current_end_frame = min(current_start_frame + segment_frames, total_frames)
            is_padded = True
        else:
            # Normal segment
            is_padded = False
        
        # Find all actions that overlap with this segment
        segment_labels = []
        
        for action in action_labels:
            action_start = action['start_frame']
            action_end = action['end_frame']
            
            # Check if action overlaps with current segment
            if action_start < current_end_frame and action_end > current_start_frame:
                # Calculate overlap boundaries within the segment
                overlap_start = max(action_start, current_start_frame)
                overlap_end = min(action_end, current_end_frame)
                
                # Calculate number of overlapping frames
                overlap_frame_count = overlap_end - overlap_start
                
                # Only include label if overlap meets minimum threshold
                if overlap_frame_count >= min_overlap_frames:
                    # Calculate coverage ratio (what fraction of segment has this action)
                    coverage_ratio = overlap_frame_count / segment_frames
                    
                    segment_labels.append({
                        'label': action['label'],
                        'original_start_frame': action_start,
                        'original_end_frame': action_end,
                        'overlap_start_frame': overlap_start,
                        'overlap_end_frame': overlap_end,
                        'overlap_frames': overlap_frame_count,
                        'coverage_ratio': coverage_ratio
                    })
        if len(segment_labels) > 0:
            # Create segment entry
            segment = {
                'segment_id': segment_id,
                'start_frame': current_start_frame,
                'end_frame': current_end_frame,
                'start_time': current_start_frame / fps,
                'end_time': current_end_frame / fps,
                'duration': (current_end_frame - current_start_frame) / fps,
                'is_padded': is_padded,
                'labels': segment_labels
            }
            
            segments.append(segment)
            segment_id += 1
        
        # Move to next segment
        current_start_frame += stride_frames
    
    return segments


def segment_actions_by_duration_v2(
    action_labels: List[Dict[str, Union[str, int]]],
    fps: float,
    total_frames: int,
    segment_duration: float = 2.0,
    overlap: float = 0.0,
    min_segment_frames: int = 30,
    min_overlap_frames: int = 15
) -> List[Dict]:
    """
    Cut action segments into fixed-duration segments aligned to action boundaries.
    
    Args:
        action_labels: List of dictionaries with keys:
            - 'label': str, action label name
            - 'start_frame': int, starting frame number
            - 'end_frame': int, ending frame number
        fps: float, frames per second of the video
        total_frames: int, total number of frames in the video
        segment_duration: float, desired segment duration in seconds (e.g., 2, 3, 4)
        overlap: float, overlap between consecutive segments in seconds (default: 0)
        min_segment_frames: int, minimum frames required for a segment (default: 30)
        min_overlap_frames: int, minimum overlap frames to include an action label (default: 15)
    
    Returns:
        List of dictionaries with segment information and overlapping action labels
    """
    if not action_labels:
        return []
    
    # Sort actions by start frame
    sorted_actions = sorted(action_labels, key=lambda x: x['start_frame'])
    
    # Calculate segment length in frames
    segment_frames = int(segment_duration * fps)
    overlap_frames = int(overlap * fps)
    stride_frames = segment_frames - overlap_frames
    
    segments = []
    segment_id = 0
    
    # Process each action
    for action in sorted_actions:
        action_start = action['start_frame']
        action_end = action['end_frame']
        action_duration_frames = action_end - action_start
        
        # Start segments aligned to the action start
        current_start_frame = action_start
        
        while current_start_frame < action_end:
            current_end_frame = current_start_frame + segment_frames
            remaining_frames = action_end - current_start_frame
            
            # Handle short remaining segments
            if remaining_frames < min_segment_frames:
                # Discard: not enough frames left
                break
            elif remaining_frames < segment_frames:
                # Pad to action end or full segment, whichever is appropriate
                current_end_frame = min(current_start_frame + segment_frames,  total_frames)
                is_padded = True
            else:
                # Normal segment
                current_end_frame = min(current_end_frame, total_frames)
                is_padded = False
            
            # Find all actions that overlap with this segment
            segment_labels = []
            
            for candidate_action in sorted_actions:
                candidate_start = candidate_action['start_frame']
                candidate_end = candidate_action['end_frame']
                
                # Check if action overlaps with current segment
                if candidate_start < current_end_frame and candidate_end > current_start_frame:
                    # Calculate overlap boundaries
                    overlap_start = max(candidate_start, current_start_frame)
                    overlap_end = min(candidate_end, current_end_frame)
                    overlap_frame_count = overlap_end - overlap_start
                    
                    # Only include label if overlap meets minimum threshold
                    if overlap_frame_count >= min_overlap_frames:
                        coverage_ratio = overlap_frame_count / segment_frames
                        
                        segment_labels.append({
                            'label': candidate_action['label'],
                            'original_start_frame': candidate_start,
                            'original_end_frame': candidate_end,
                            'overlap_start_frame': overlap_start,
                            'overlap_end_frame': overlap_end,
                            'overlap_frames': overlap_frame_count,
                            'coverage_ratio': coverage_ratio
                        })
            
            if len(segment_labels) > 0:
                # Create segment entry
                segment = {
                    'segment_id': segment_id,
                    'start_frame': current_start_frame,
                    'end_frame': current_end_frame,
                    'start_time': current_start_frame / fps,
                    'end_time': current_end_frame / fps,
                    'duration': (current_end_frame - current_start_frame) / fps,
                    'is_padded': is_padded,
                    'labels': segment_labels
                }
                
                segments.append(segment)
                segment_id += 1
            
            # Move to next segment with stride
            current_start_frame += stride_frames
    
    remove_duplicate_segments = []
    unique_segment_keys = set()
    for seg in segments:
        key = (seg['start_frame'], seg['end_frame'])
        if key not in unique_segment_keys:
            unique_segment_keys.add(key)
            remove_duplicate_segments.append(seg)
    return remove_duplicate_segments


def get_dominant_label(segment: Dict) -> str:
    """
    Get the dominant label for a segment based on coverage ratio.
    
    Args:
        segment: A segment dictionary from segment_actions_by_duration
    
    Returns:
        The label with the highest coverage ratio, or 'background' if no labels
    """
    if not segment['labels']:
        return 'others'
    
    # Find label with maximum coverage
    dominant = max(segment['labels'], key=lambda x: x['coverage_ratio'])
    return dominant['label']


# Example usage
if __name__ == "__main__":
    # Example action labels
    actions = [
        {'label': 'walking', 'start_frame': 10, 'end_frame': 150},
        {'label': 'jumping', 'start_frame': 100, 'end_frame': 145},
        # {'label': 'running', 'start_frame': 300, 'end_frame': 400},
        # {'label': 'waving', 'start_frame': 405, 'end_frame': 412},  # Only 7 frames - should be filtered
    ]
    
    fps = 30  # 30 frames per second
    total_frames = 155  # Video has 425 frames total
    
    # Test with 2-second segments
    print("=" * 80)
    print("2-SECOND SEGMENTS (60 frames)")
    print(f"Total frames: {total_frames}, Min segment frames: 30, Min overlap frames: 15")
    print("=" * 80)
    segments_2s = segment_actions_by_duration_v2(
        actions, fps, total_frames, 
        segment_duration=2.0,
        min_segment_frames=30,
        min_overlap_frames=15
    )
    
    for seg in segments_2s:
        print(f"\nSegment {seg['segment_id']}:")
        print(f"  Frames: {seg['start_frame']} - {seg['end_frame']} (Padded: {seg['is_padded']})")
        print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        print(f"  Labels in this segment:")
        if seg['labels']:
            for lbl in seg['labels']:
                print(f"    - {lbl['label']}: {lbl['overlap_frames']} frames ({lbl['coverage_ratio']:.2%} coverage)")
        else:
            print(f"    - (no labels with >= 12 frame overlap)")
        print(f"  Dominant label: {get_dominant_label(seg)}")
    
    # Test with 3-second segments
    print("\n" + "=" * 80)
    print("3-SECOND SEGMENTS (90 frames)")
    print("=" * 80)
    segments_3s = segment_actions_by_duration_v2(
        actions, fps, total_frames,
        segment_duration=3.0,
        min_segment_frames=30,
        min_overlap_frames=15
    )
    
    for seg in segments_3s:
        print(f"\nSegment {seg['segment_id']}:")
        print(f"  Frames: {seg['start_frame']} - {seg['end_frame']} (Padded: {seg['is_padded']})")
        print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        if seg['labels']:
            for lbl in seg['labels']:
                print(f"    - {lbl['label']}: {lbl['overlap_frames']} frames ({lbl['coverage_ratio']:.2%} coverage)")
        else:
            print(f"    - (no labels with >= 12 frame overlap)")
        print(f"  Dominant label: {get_dominant_label(seg)}")
    
    # Test with 4-second segments and 1-second overlap
    print("\n" + "=" * 80)
    print("4-SECOND SEGMENTS WITH 1-SECOND OVERLAP (120 frames, 30 frame stride)")
    print("=" * 80)
    segments_4s = segment_actions_by_duration_v2(
        actions, fps, total_frames,
        segment_duration=4.0,
        overlap=1.0,
        min_segment_frames=30,
        min_overlap_frames=15
    )
    
    for seg in segments_4s:
        print(f"\nSegment {seg['segment_id']}:")
        print(f"  Frames: {seg['start_frame']} - {seg['end_frame']} (Padded: {seg['is_padded']})")
        print(f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        if seg['labels']:
            for lbl in seg['labels']:
                print(f"    - {lbl['label']}: {lbl['overlap_frames']} frames ({lbl['coverage_ratio']:.2%} coverage)")
        else:
            print(f"    - (no labels with >= 12 frame overlap)")
        print(f"  Dominant label: {get_dominant_label(seg)}")
    
    # Demonstrate the min_segment_frames threshold
    print("\n" + "=" * 80)
    print("DEMONSTRATING MIN_SEGMENT_FRAMES THRESHOLD")
    print("=" * 80)
    print(f"Video ends at frame {total_frames}")
    print(f"Last full segment would start at frame {segments_2s[-1]['start_frame'] if segments_2s else 'N/A'}")
    print(f"Frames remaining after last segment: {total_frames - (segments_2s[-1]['start_frame'] + 60) if segments_2s else 'N/A'}")
    print(f"Since remaining frames < 30, the last partial segment was DISCARDED")