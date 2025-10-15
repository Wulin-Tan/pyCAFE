"""Core frame extraction functions with K-means clustering"""

import time
from .extraction import extract_downsampled_frames_gpu, extract_specific_frames
from .clustering import perform_kmeans_gpu
from .utils import get_video_info


def extract_frames_kmeans_gpu(
    video_path,
    output_dir,
    n_frames=10,
    start_time=0.0,
    end_time=1.0,
    step=1,
    resize_width=30,
    use_color=False,
    device_id=0,
    max_iter=100,
    tol=1e-4,
    random_state=42,
    chunk_size=None
):
    """
    Extract frames using GPU-accelerated K-means clustering
    
    This is the main function for intelligent frame extraction. It:
    1. Extracts downsampled frames on GPU
    2. Performs K-means clustering to find representative frames
    3. Extracts selected frames at full resolution
    
    Args:
        video_path (str): Path to video file
        output_dir (str): Output directory for extracted frames
        n_frames (int, optional): Number of frames to extract. Default: 10
        start_time (float, optional): Start time as fraction (0.0-1.0). Default: 0.0
        end_time (float, optional): End time as fraction (0.0-1.0). Default: 1.0
        step (int, optional): Sample every nth frame. Default: 1
        resize_width (int, optional): Width for clustering thumbnails. Default: 30
        use_color (bool, optional): Use RGB (True) or grayscale (False). Default: False
        device_id (int, optional): GPU device ID. Default: 0
        max_iter (int, optional): Maximum K-means iterations. Default: 100
        tol (float, optional): K-means convergence tolerance. Default: 1e-4
        random_state (int, optional): Random seed. Default: 42
        chunk_size (int, optional): Frames per chunk (auto if None). Default: None
    
    Returns:
        tuple: (saved_files, timing_dict)
            - saved_files: List of paths to extracted frames
            - timing_dict: Dictionary with timing information
    
    Example:
        >>> files, timing = extract_frames_kmeans_gpu(
        ...     "video.mp4",
        ...     "./output",
        ...     n_frames=50,
        ...     step=5
        ... )
        >>> print(f"Extracted {len(files)} frames in {timing['total_time']:.2f}s")
    
    Notes:
        - Automatically handles videos longer than available GPU memory
        - Progress bars show extraction status
        - Returns timing information for benchmarking
    """
    
    # Get video info
    video_info = get_video_info(video_path)
    total_frames = video_info['nframes']
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    print(f"\n{'='*70}")
    print(f"VIDEO INFORMATION (GPU MODE)")
    print(f"{'='*70}")
    print(f"Path: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {video_info['duration']:.2f}s")
    print(f"Resolution: {width}x{height}")
    
    # Calculate frame range
    start_frame = int(total_frames * start_time)
    end_frame = int(total_frames * end_time)
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION PARAMETERS")
    print(f"{'='*70}")
    print(f"Frame range: {start_frame} - {end_frame}")
    print(f"Sampling step: {step}")
    print(f"Frames to extract: {n_frames}")
    print(f"Clustering resize: {resize_width}px")
    print(f"Use color: {use_color}")
    if chunk_size:
        print(f"Chunk size: {chunk_size} (manual)")
    else:
        print(f"Chunk size: auto")
    
    # Step 1: Extract downsampled frames on GPU
    print(f"\n{'='*70}")
    print(f"STEP 1: EXTRACTING FRAMES (GPU - NVIDIA DALI)")
    print(f"{'='*70}")
    
    t1_start = time.time()
    
    frames_data, frame_indices = extract_downsampled_frames_gpu(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        step=step,
        resize_width=resize_width,
        use_color=use_color,
        device_id=device_id,
        chunk_size=chunk_size
    )
    
    t1_end = time.time()
    t1_duration = t1_end - t1_start
    
    print(f"✅ Extracted frames in {t1_duration:.2f}s")
    
    if frames_data is None or len(frames_data) == 0:
        print("❌ No frames extracted!")
        return [], None
    
    # Ensure we don't request more frames than available
    n_frames_actual = min(n_frames, len(frames_data))
    if n_frames_actual < n_frames:
        print(f"⚠️  Requested {n_frames} frames but only {n_frames_actual} available")
    
    # Step 2: Perform k-means clustering
    print(f"\n{'='*70}")
    print(f"STEP 2: K-MEANS CLUSTERING (GPU - cuML)")
    print(f"{'='*70}")
    
    t2_start = time.time()
    
    selected_indices = perform_kmeans_gpu(
        frames_data=frames_data,
        n_clusters=n_frames_actual,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    
    t2_end = time.time()
    t2_duration = t2_end - t2_start
    
    print(f"✅ K-means completed in {t2_duration:.2f}s")
    
    # Step 3: Extract full-resolution frames
    print(f"\n{'='*70}")
    print(f"STEP 3: EXTRACTING FULL-RESOLUTION FRAMES")
    print(f"{'='*70}")
    
    t3_start = time.time()
    
    selected_frame_numbers = [frame_indices[i] for i in selected_indices]
    
    saved_files = extract_specific_frames(
        video_path=video_path,
        frame_indices=selected_frame_numbers,
        output_dir=output_dir
    )
    
    t3_end = time.time()
    t3_duration = t3_end - t3_start
    
    print(f"✅ Extracted full-res frames in {t3_duration:.2f}s")
    
    # Summary
    total_time = t1_duration + t2_duration + t3_duration
    
    print(f"\n{'='*70}")
    print(f"✅ GPU EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Step 1 (Frame extraction): {t1_duration:>8.2f}s ({t1_duration/total_time*100:>5.1f}%)")
    print(f"Step 2 (K-means):          {t2_duration:>8.2f}s ({t2_duration/total_time*100:>5.1f}%)")
    print(f"Step 3 (Full-res extract): {t3_duration:>8.2f}s ({t3_duration/total_time*100:>5.1f}%)")
    print(f"{'-'*70}")
    print(f"TOTAL TIME:                {total_time:>8.2f}s")
    print(f"Extracted {len(saved_files)} frames")
    print(f"Output directory: {output_dir}")
    
    return saved_files, {
        'step1_time': t1_duration,
        'step2_time': t2_duration,
        'step3_time': t3_duration,
        'total_time': total_time
    }


def extract_frames_kmeans_cpu(
    video_path,
    output_dir,
    n_frames=10,
    start_time=0.0,
    end_time=1.0,
    step=1,
    resize_width=30,
    use_color=False,
    max_iter=100,
    tol=1e-4,
    random_state=42
):
    """
    Extract frames using CPU-only K-means clustering
    
    CPU-only version for comparison and fallback. Uses OpenCV for video
    reading and scikit-learn for K-means clustering.
    
    Args:
        video_path (str): Path to video file
        output_dir (str): Output directory for extracted frames
        n_frames (int, optional): Number of frames to extract. Default: 10
        start_time (float, optional): Start time as fraction (0.0-1.0). Default: 0.0
        end_time (float, optional): End time as fraction (0.0-1.0). Default: 1.0
        step (int, optional): Sample every nth frame. Default: 1
        resize_width (int, optional): Width for clustering thumbnails. Default: 30
        use_color (bool, optional): Use RGB (True) or grayscale (False). Default: False
        max_iter (int, optional): Maximum K-means iterations. Default: 100
        tol (float, optional): K-means convergence tolerance. Default: 1e-4
        random_state (int, optional): Random seed. Default: 42
    
    Returns:
        tuple: (saved_files, timing_dict)
            - saved_files: List of paths to extracted frames
            - timing_dict: Dictionary with timing information
    
    Example:
        >>> files, timing = extract_frames_kmeans_cpu(
        ...     "video.mp4",
        ...     "./output",
        ...     n_frames=50,
        ...     step=5
        ... )
        >>> print(f"Extracted {len(files)} frames in {timing['total_time']:.2f}s")
    """
    
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans
    from tqdm import tqdm
    
    # Get video info
    video_info = get_video_info(video_path)
    total_frames = video_info['nframes']
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    
    print(f"\n{'='*70}")
    print(f"VIDEO INFORMATION (CPU MODE)")
    print(f"{'='*70}")
    print(f"Path: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {video_info['duration']:.2f}s")
    print(f"Resolution: {width}x{height}")
    
    # Calculate frame range
    start_frame = int(total_frames * start_time)
    end_frame = int(total_frames * end_time)
    
    print(f"\n{'='*70}")
    print(f"EXTRACTION PARAMETERS")
    print(f"{'='*70}")
    print(f"Frame range: {start_frame} - {end_frame}")
    print(f"Sampling step: {step}")
    print(f"Frames to extract: {n_frames}")
    print(f"Clustering resize: {resize_width}px")
    print(f"Use color: {use_color}")
    
    # Step 1: Extract and downsample frames (CPU)
    print(f"\n{'='*70}")
    print(f"STEP 1: EXTRACTING FRAMES (CPU - OpenCV)")
    print(f"{'='*70}")
    
    t1_start = time.time()
    
    cap = cv2.VideoCapture(video_path)
    frame_indices = list(range(start_frame, end_frame, step))
    frames_list = []
    
    for frame_idx in tqdm(frame_indices, desc="Reading frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Resize
        aspect_ratio = frame.shape[0] / frame.shape[1]
        new_height = int(resize_width * aspect_ratio)
        resized = cv2.resize(frame, (resize_width, new_height))
        
        # Color or grayscale
        if use_color:
            # Flatten RGB
            flat_frame = resized.reshape(-1)
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            flat_frame = gray.reshape(-1)
        
        frames_list.append(flat_frame)
    
    cap.release()
    
    frames_data = np.array(frames_list, dtype=np.float32)
    
    t1_end = time.time()
    t1_duration = t1_end - t1_start
    
    print(f"✅ Extracted {len(frames_data)} frames in {t1_duration:.2f}s")
    print(f"   Speed: {len(frames_data)/t1_duration:.1f} frames/s")
    
    # Ensure we don't request more frames than available
    n_frames_actual = min(n_frames, len(frames_data))
    if n_frames_actual < n_frames:
        print(f"⚠️  Requested {n_frames} frames but only {n_frames_actual} available")
    
    # Step 2: Perform k-means clustering (CPU)
    print(f"\n{'='*70}")
    print(f"STEP 2: K-MEANS CLUSTERING (CPU - scikit-learn)")
    print(f"{'='*70}")
    
    t2_start = time.time()
    
    print(f"Clustering data shape: {frames_data.shape}")
    print(f"Number of clusters: {n_frames_actual}")
    
    kmeans = KMeans(
        n_clusters=n_frames_actual,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        n_init=10
    )
    
    labels = kmeans.fit_predict(frames_data)
    centers = kmeans.cluster_centers_
    
    t2_end = time.time()
    t2_duration = t2_end - t2_start
    
    print(f"✅ K-means converged in {t2_duration:.2f}s")
    print(f"   Iterations: {kmeans.n_iter_}")
    
    # Select one frame per cluster
    selected_indices = []
    for cluster_id in range(n_frames_actual):
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        cluster_frames = frames_data[cluster_indices]
        center = centers[cluster_id]
        distances = np.linalg.norm(cluster_frames - center, axis=1)
        closest_idx = np.argmin(distances)
        original_idx = cluster_indices[closest_idx]
        selected_indices.append(original_idx)
    
    selected_indices = sorted(selected_indices)
    
    # Step 3: Extract full-resolution frames
    print(f"\n{'='*70}")
    print(f"STEP 3: EXTRACTING FULL-RESOLUTION FRAMES")
    print(f"{'='*70}")
    
    t3_start = time.time()
    
    selected_frame_numbers = [frame_indices[i] for i in selected_indices]
    saved_files = extract_specific_frames(
        video_path=video_path,
        frame_indices=selected_frame_numbers,
        output_dir=output_dir
    )
    
    t3_end = time.time()
    t3_duration = t3_end - t3_start
    
    print(f"✅ Extracted full-res frames in {t3_duration:.2f}s")
    
    # Summary
    total_time = t1_duration + t2_duration + t3_duration
    
    print(f"\n{'='*70}")
    print(f"✅ CPU EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Step 1 (Frame extraction): {t1_duration:>8.2f}s ({t1_duration/total_time*100:>5.1f}%)")
    print(f"Step 2 (K-means):          {t2_duration:>8.2f}s ({t2_duration/total_time*100:>5.1f}%)")
    print(f"Step 3 (Full-res extract): {t3_duration:>8.2f}s ({t3_duration/total_time*100:>5.1f}%)")
    print(f"{'-'*70}")
    print(f"TOTAL TIME:                {total_time:>8.2f}s")
    print(f"Extracted {len(saved_files)} frames")
    print(f"Output directory: {output_dir}")
    
    return saved_files, {
        'step1_time': t1_duration,
        'step2_time': t2_duration,
        'step3_time': t3_duration,
        'total_time': total_time
    }
