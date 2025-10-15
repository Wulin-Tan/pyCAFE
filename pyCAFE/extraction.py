"""Frame extraction functions using NVIDIA DALI and OpenCV"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

try:
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from .pipelines import clustering_pipeline
from .utils import get_video_info


def extract_downsampled_frames_gpu(
    video_path,
    start_frame,
    end_frame,
    step=1,
    resize_width=30,
    use_color=False,
    batch_size=256,
    device_id=0,
    chunk_size=None
):
    """
    Extract and downsample frames entirely on GPU
    Uses chunk-based processing to prevent slowdown
    
    Args:
        video_path (str): Path to video file
        start_frame (int): Start frame index
        end_frame (int): End frame index (exclusive)
        step (int, optional): Sample every nth frame. Default: 1
        resize_width (int, optional): Width to resize frames to. Default: 30
        use_color (bool, optional): Keep RGB (True) or convert to grayscale (False). Default: False
        batch_size (int, optional): Frames per DALI batch. Default: 256
        device_id (int, optional): GPU device ID. Default: 0
        chunk_size (int, optional): Frames per chunk (auto if None). Default: None
    
    Returns:
        tuple: (frames_gpu, frame_indices)
            - frames_gpu: CuPy array on GPU (if available) or NumPy array
            - frame_indices: List of extracted frame indices
    
    Example:
        >>> frames, indices = extract_downsampled_frames_gpu(
        ...     "video.mp4", 0, 10000, step=5, resize_width=30
        ... )
        >>> print(f"Extracted {len(frames)} frames")
    
    Notes:
        - Uses chunk-based processing to prevent memory issues
        - Automatically determines optimal chunk size if not specified
        - Shows progress bar for each chunk
    """
    
    frame_indices = list(range(start_frame, end_frame, step))
    n_frames = len(frame_indices)
    
    if n_frames == 0:
        return None, []
    
    print(f"\n📹 Extracting {n_frames} downsampled frames on GPU...")
    print(f"   Range: {start_frame}-{end_frame}, step: {step}")
    print(f"   Resize: {resize_width}, Color: {use_color}")
    
    # Determine chunk size (auto or manual)
    if chunk_size is None:
        # Automatic chunk sizing based on total frames
        if n_frames < 2000:
            chunk_size = n_frames  # No chunking needed
        elif n_frames < 10000:
            chunk_size = 2000
        else:
            chunk_size = 3000
        print(f"   Auto chunk size: {chunk_size}")
    else:
        print(f"   Manual chunk size: {chunk_size}")
    
    n_chunks = (n_frames + chunk_size - 1) // chunk_size
    
    if n_chunks > 1:
        print(f"   Processing in {n_chunks} chunks")
        print(f"   (Prevents slowdown by rebuilding pipeline)\n")
    
    all_chunk_arrays = []
    
    for chunk_idx in range(n_chunks):
        # Calculate chunk boundaries
        chunk_start_idx = chunk_idx * chunk_size
        chunk_end_idx = min((chunk_idx + 1) * chunk_size, n_frames)
        chunk_n_frames = chunk_end_idx - chunk_start_idx
        
        chunk_start_frame = frame_indices[chunk_start_idx]
        chunk_end_frame = frame_indices[chunk_end_idx - 1] + step if chunk_end_idx > chunk_start_idx else chunk_start_frame
        
        # Calculate sequence length for this chunk
        sequence_length = min(batch_size, max(32, chunk_n_frames // 5))
        
        # Build FRESH pipeline for this chunk
        pipe = clustering_pipeline(
            video_path=video_path,
            sequence_length=sequence_length,
            step=step,
            resize_width=resize_width,
            batch_size=1,
            num_threads=4,
            device_id=device_id,
            prefetch_queue_depth=2
        )
        pipe.build()
        
        # Skip to chunk start frame
        frames_to_skip = chunk_start_frame // step
        batches_to_skip = frames_to_skip // sequence_length
        
        if batches_to_skip > 0:
            for _ in range(batches_to_skip):
                try:
                    _ = pipe.run()
                except StopIteration:
                    break
        
        # Extract frames for this chunk
        chunk_frames_list = []
        frames_extracted_in_chunk = 0
        
        # Single progress bar for this chunk
        chunk_pbar = tqdm(
            total=chunk_n_frames, 
            desc=f"Chunk {chunk_idx + 1}/{n_chunks}",
            unit="frames",
            leave=True,
            ncols=100,
            miniters=1,
            mininterval=0.5
        )
        
        try:
            while frames_extracted_in_chunk < chunk_n_frames:
                outputs = pipe.run()
                frames_gpu_batch = outputs[0]
                
                # Move to CPU
                frames_cpu = frames_gpu_batch.as_cpu()
                frames_np = frames_cpu.as_array()[0]  # [seq_len, H, W, 3]
                
                actual_batch_size = frames_np.shape[0]
                frames_to_process = min(actual_batch_size, chunk_n_frames - frames_extracted_in_chunk)
                
                # Vectorized processing
                if use_color:
                    # Keep all 3 channels, concatenate horizontally
                    batch_processed = np.concatenate([
                        frames_np[:frames_to_process, :, :, 0],
                        frames_np[:frames_to_process, :, :, 1],
                        frames_np[:frames_to_process, :, :, 2]
                    ], axis=2)
                else:
                    # Convert to grayscale
                    batch_processed = np.mean(frames_np[:frames_to_process], axis=3, dtype=np.float32)
                
                # Add to chunk list
                for i in range(frames_to_process):
                    chunk_frames_list.append(batch_processed[i])
                
                frames_extracted_in_chunk += frames_to_process
                chunk_pbar.update(frames_to_process)
                
                # Clean up batch
                del frames_np, frames_cpu, frames_gpu_batch, batch_processed
                
                if frames_extracted_in_chunk >= chunk_n_frames:
                    break
                    
        except StopIteration:
            chunk_pbar.write(f"   ⚠️  Reached end at {frames_extracted_in_chunk} frames")
        finally:
            chunk_pbar.close()
            # Critical: Delete pipeline after each chunk
            del pipe
            import gc
            gc.collect()
        
        # Convert chunk to array
        if chunk_frames_list:
            chunk_array = np.array(chunk_frames_list, dtype=np.float32)
            all_chunk_arrays.append(chunk_array)
            del chunk_frames_list
            
            print(f"   ✅ Chunk {chunk_idx + 1}/{n_chunks} complete: {len(chunk_array)} frames\n")
    
    # Combine all chunks
    if len(all_chunk_arrays) > 1:
        print("🔗 Combining chunks...")
        frames_data = np.concatenate(all_chunk_arrays, axis=0)
        del all_chunk_arrays
    else:
        frames_data = all_chunk_arrays[0]
    
    # Convert to GPU array if cuML available
    if CUML_AVAILABLE:
        print("↗️  Moving data to GPU (CuPy array)")
        frames_gpu = cp.asarray(frames_data)
        print(f"GPU memory used: {frames_gpu.nbytes / 1e9:.2f} GB")
        del frames_data
        import gc
        gc.collect()
    else:
        print("⚠️  Keeping on CPU (NumPy array)")
        frames_gpu = frames_data
    
    print(f"\n✅ Extracted {len(frames_gpu)} frames")
    print(f"   Shape: {frames_gpu.shape}")
    
    return frames_gpu, frame_indices[:len(frames_gpu)]


def extract_specific_frames(video_path, frame_indices, output_dir):
    """
    Extract specific frames from video at full resolution using OpenCV
    
    Args:
        video_path (str): Path to video file
        frame_indices (list): List of frame indices to extract
        output_dir (str): Output directory for saved frames
    
    Returns:
        list: Paths to saved frame files
    
    Example:
        >>> saved = extract_specific_frames("video.mp4", [10, 50, 100], "./frames")
        >>> print(f"Saved {len(saved)} frames")
    
    Notes:
        - Uses OpenCV for reliable random frame access
        - Saves frames as PNG files
        - Handles missing frames gracefully
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return []
    
    saved_files = []
    
    print(f"Extracting {len(frame_indices)} frames at full resolution...")
    
    for idx, frame_num in enumerate(tqdm(frame_indices, desc="Extracting frames")):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print(f"⚠️  Failed to read frame {frame_num}")
            continue
        
        # Save frame
        output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.png")
        cv2.imwrite(output_path, frame)
        saved_files.append(output_path)
    
    cap.release()
    
    print(f"✅ Saved {len(saved_files)} frames to {output_dir}")
    
    return saved_files


def extract_and_save_fullres_frames(
    video_path,
    frame_indices,
    output_dir,
    file_prefix="frame",
    crop_box=None,
    device_id=0
):
    """
    Extract full-resolution frames and save them
    
    Args:
        video_path (str): Path to video file
        frame_indices (list): Frame indices to extract
        output_dir (str): Output directory
        file_prefix (str, optional): Filename prefix. Default: "frame"
        crop_box (tuple, optional): Crop box as (x1, y1, x2, y2). Default: None
        device_id (int, optional): GPU device ID (unused, for API compatibility). Default: 0
    
    Returns:
        list: Paths to saved frame files
    
    Example:
        >>> saved = extract_and_save_fullres_frames(
        ...     "video.mp4", [10, 50, 100], "./frames", crop_box=(0, 0, 640, 480)
        ... )
    
    Notes:
        - Converts BGR to RGB before saving
        - Supports optional cropping
        - Uses OpenCV for extraction (reliable random access)
    """
    
    print("\n" + "="*70)
    print("💾 EXTRACTING & SAVING FULL RESOLUTION FRAMES")
    print("="*70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Output: {output_path}")
    print(f"🎬 Processing {len(frame_indices)} frames...")
    
    # Get video info for filename padding
    video_info = get_video_info(video_path)
    padding = len(str(video_info['nframes']))
    
    saved_files = []
    
    # Use OpenCV for random frame access (most reliable method)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    print("\n📹 Extracting frames at full resolution...")
    
    for frame_idx in tqdm(frame_indices, desc="Extracting & saving"):
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crop if needed
            if crop_box:
                x1, y1, x2, y2 = crop_box
                frame_rgb = frame_rgb[y1:y2, x1:x2, :]
            
            # Save immediately (minimizes memory usage)
            img = Image.fromarray(frame_rgb)
            filename = f"{file_prefix}_{str(frame_idx).zfill(padding)}.png"
            filepath = output_path / filename
            img.save(filepath)
            
            saved_files.append(str(filepath))
        else:
            print(f"\n⚠️ Failed to read frame {frame_idx}")
    
    cap.release()
    
    print(f"\n✅ Saved {len(saved_files)}/{len(frame_indices)} frames")
    print(f"📁 Location: {output_path}")
    
    return saved_files
