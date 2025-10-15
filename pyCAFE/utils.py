"""Utility functions for video processing"""

import cv2


def get_video_info(video_path):
    """
    Get video metadata using OpenCV
    
    Args:
        video_path (str): Path to video file
    
    Returns:
        dict: Dictionary containing:
            - nframes (int): Total number of frames
            - fps (float): Frames per second
            - width (int): Frame width in pixels
            - height (int): Frame height in pixels
            - duration (float): Video duration in seconds
    
    Raises:
        ValueError: If video cannot be opened
    
    Example:
        >>> info = get_video_info("video.mp4")
        >>> print(f"Video has {info['nframes']} frames at {info['fps']} FPS")
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    info = {
        'nframes': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    info['duration'] = info['nframes'] / info['fps'] if info['fps'] > 0 else 0
    
    cap.release()
    
    return info


def format_time(seconds):
    """
    Format seconds into human-readable time string
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    
    Example:
        >>> format_time(125.5)
        '2m 5.5s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def validate_time_range(start_time, end_time):
    """
    Validate start and end time parameters
    
    Args:
        start_time (float): Start time (0.0-1.0)
        end_time (float): End time (0.0-1.0)
    
    Raises:
        ValueError: If time range is invalid
    """
    if not (0.0 <= start_time <= 1.0):
        raise ValueError(f"start_time must be between 0.0 and 1.0, got {start_time}")
    if not (0.0 <= end_time <= 1.0):
        raise ValueError(f"end_time must be between 0.0 and 1.0, got {end_time}")
    if start_time >= end_time:
        raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")
