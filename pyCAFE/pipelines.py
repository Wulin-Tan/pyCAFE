"""NVIDIA DALI pipeline definitions for GPU video processing"""

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def clustering_pipeline(video_path, sequence_length, step, resize_width):
    """
    GPU pipeline for extracting downsampled frames for clustering
    
    This pipeline efficiently decodes and downsamples video frames on GPU
    for use in K-means clustering to select representative frames.
    
    Args:
        video_path (str): Path to video file
        sequence_length (int): Number of frames to decode per batch
        step (int): Frame sampling step (e.g., 5 = every 5th frame)
        resize_width (int): Target width for downsampling
    
    Returns:
        Tensor: Downsampled frames on GPU with shape [batch, height, width, 3]
    
    Notes:
        - Uses GPU video reader for hardware-accelerated decoding
        - Resizes frames proportionally based on resize_width
        - Does not pad incomplete batches to avoid extra decoding
    """
    video = fn.readers.video(
        device="gpu",
        filenames=[video_path],
        sequence_length=sequence_length,
        step=step,
        random_shuffle=False,
        pad_last_batch=False,
        file_list_include_preceding_frame=False,
        skip_vfr_check=True,
        dtype=types.UINT8
    )
    
    # Resize on GPU - maintains aspect ratio
    resized = fn.resize(
        video,
        device="gpu",
        resize_shorter=resize_width,
        interp_type=types.INTERP_LINEAR
    )
    
    return resized


@pipeline_def
def fullres_extraction_pipeline(video_path, frame_indices, batch_size):
    """
    GPU pipeline for extracting specific frames at full resolution
    
    This pipeline is used to extract the final selected frames at their
    original resolution after clustering analysis.
    
    Args:
        video_path (str): Path to video file
        frame_indices (list): List of specific frame indices to extract
        batch_size (int): Number of frames per batch
    
    Returns:
        Tensor: Full resolution frames on GPU
    
    Notes:
        - Extracts frames at original resolution
        - More efficient than seeking with OpenCV for multiple frames
        - Currently not fully implemented - OpenCV fallback used in practice
    """
    video = fn.readers.video(
        device="gpu",
        filenames=[video_path],
        sequence_length=1,
        random_shuffle=False,
        pad_last_batch=False,
        initial_fill=batch_size
    )
    
    return video
