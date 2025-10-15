"""K-means clustering for intelligent frame selection"""

import numpy as np

try:
    from cuml.cluster import KMeans as cuKMeans
    import cupy as cp
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.cluster import KMeans
    CUML_AVAILABLE = False


def perform_kmeans_gpu(
    frames_data,
    n_clusters,
    max_iter=100,
    tol=1e-4,
    random_state=42
):
    """
    Perform k-means clustering to select representative frames
    
    Uses GPU-accelerated K-means (cuML) if available, otherwise falls back
    to CPU implementation (scikit-learn). Selects one representative frame
    per cluster (the frame closest to the cluster center).
    
    Args:
        frames_data (array): Frame data as CuPy array (GPU) or NumPy array (CPU)
            Shape: (n_frames, height, width) for grayscale or
                   (n_frames, height, width, 3) for color
        n_clusters (int): Number of clusters (frames to select)
        max_iter (int, optional): Maximum K-means iterations. Default: 100
        tol (float, optional): Convergence tolerance. Default: 1e-4
        random_state (int, optional): Random seed for reproducibility. Default: 42
    
    Returns:
        list: Indices of selected frames (one per cluster), sorted by frame index
    
    Example:
        >>> frames = extract_downsampled_frames_gpu(...)
        >>> selected = perform_kmeans_gpu(frames, n_clusters=10)
        >>> print(f"Selected frames: {selected}")
        Selected frames: [45, 120, 234, 456, ...]
    
    Notes:
        - Automatically flattens frame data for clustering
        - Returns all frames if fewer frames than clusters requested
        - Selects temporally distributed frames via clustering
    """
    
    n_samples = len(frames_data)
    
    # Handle edge case: fewer frames than clusters
    if n_samples < n_clusters:
        print(f"⚠️  Only {n_samples} frames available, selecting all")
        return list(range(n_samples))
    
    # Flatten frames for clustering
    if len(frames_data.shape) == 3:
        # Grayscale: (n_frames, height, width) -> (n_frames, height*width)
        frames_flat = frames_data.reshape(n_samples, -1)
    elif len(frames_data.shape) == 4:
        # Color: (n_frames, height, width, 3) -> (n_frames, height*width*3)
        frames_flat = frames_data.reshape(n_samples, -1)
    else:
        raise ValueError(f"Unexpected frame data shape: {frames_data.shape}")
    
    print(f"Clustering data shape: {frames_flat.shape}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Max iterations: {max_iter}")
    
    # Perform k-means clustering
    if CUML_AVAILABLE and isinstance(frames_data, cp.ndarray):
        print("Using cuML KMeans (GPU acceleration)")
        
        kmeans = cuKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            verbose=0
        )
        
        # Fit and predict on GPU
        print("Fitting k-means on GPU...")
        labels = kmeans.fit_predict(frames_flat)
        cluster_centers = kmeans.cluster_centers_
        
        # Convert to CPU for distance calculation
        labels_cpu = cp.asnumpy(labels)
        frames_flat_cpu = cp.asnumpy(frames_flat)
        centers_cpu = cp.asnumpy(cluster_centers)
        
    else:
        print("Using scikit-learn KMeans (CPU)")
        from sklearn.cluster import KMeans
        
        # Convert to NumPy if needed
        if isinstance(frames_flat, cp.ndarray):
            frames_flat_cpu = cp.asnumpy(frames_flat)
        else:
            frames_flat_cpu = frames_flat
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=10
        )
        
        print("Fitting k-means on CPU...")
        labels_cpu = kmeans.fit_predict(frames_flat_cpu)
        centers_cpu = kmeans.cluster_centers_
    
    print(f"✅ K-means converged")
    print(f"   Iterations: {kmeans.n_iter_ if hasattr(kmeans, 'n_iter_') else 'N/A'}")
    
    # Select one frame per cluster (closest to center)
    print("\nSelecting representative frames...")
    selected_indices = []
    
    for cluster_id in range(n_clusters):
        # Find all frames in this cluster
        cluster_mask = (labels_cpu == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            print(f"⚠️  Cluster {cluster_id} is empty, skipping")
            continue
        
        # Get frames in this cluster
        cluster_frames = frames_flat_cpu[cluster_indices]
        
        # Find frame closest to cluster center
        center = centers_cpu[cluster_id]
        distances = np.linalg.norm(cluster_frames - center, axis=1)
        closest_idx_in_cluster = np.argmin(distances)
        
        # Map back to original frame index
        original_idx = cluster_indices[closest_idx_in_cluster]
        selected_indices.append(original_idx)
    
    # Sort by frame index (temporal order)
    selected_indices = sorted(selected_indices)
    
    print(f"✅ Selected {len(selected_indices)} representative frames")
    
    # Show cluster distribution statistics
    unique, counts = np.unique(labels_cpu, return_counts=True)
    print(f"\nCluster size distribution:")
    print(f"   Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.1f}")
    
    return selected_indices
