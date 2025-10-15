"""Benchmarking utilities for CPU vs GPU comparison"""

import time
from .core import extract_frames_kmeans_gpu, extract_frames_kmeans_cpu


def benchmark_cpu_vs_gpu(
    video_path,
    output_dir_gpu="./gpu_output",
    output_dir_cpu="./cpu_output",
    **kwargs
):
    """
    Benchmark CPU vs GPU frame extraction performance
    
    Runs both GPU and CPU extraction with identical parameters and
    compares the performance.
    
    Args:
        video_path (str): Path to video file
        output_dir_gpu (str, optional): GPU output directory. Default: "./gpu_output"
        output_dir_cpu (str, optional): CPU output directory. Default: "./cpu_output"
        **kwargs: Additional parameters passed to extraction functions
            (n_frames, start_time, end_time, step, resize_width, etc.)
    
    Returns:
        dict: Benchmark results containing:
            - cpu_timing: CPU timing breakdown
            - gpu_timing: GPU timing breakdown
            - speedup: Overall speedup factor
            - cpu_files: List of CPU-extracted files
            - gpu_files: List of GPU-extracted files
    
    Example:
        >>> results = benchmark_cpu_vs_gpu(
        ...     "video.mp4",
        ...     n_frames=50,
        ...     step=5
        ... )
        >>> print(f"GPU is {results['speedup']:.2f}x faster")
    
    Notes:
        - Creates separate output directories for CPU and GPU results
        - Shows detailed timing breakdown for each step
        - Useful for validating GPU acceleration benefits
    """
    
    print("="*70)
    print("🔥 BENCHMARK: CPU vs GPU")
    print("="*70)
    
    # GPU test
    print("\n🚀 GPU Test...")
    gpu_start = time.time()
    gpu_files, gpu_timing = extract_frames_kmeans_gpu(
        video_path=video_path,
        output_dir=output_dir_gpu,
        **kwargs
    )
    gpu_total = time.time() - gpu_start
    
    # CPU test
    print("\n💻 CPU Test...")
    cpu_start = time.time()
    cpu_files, cpu_timing = extract_frames_kmeans_cpu(
        video_path=video_path,
        output_dir=output_dir_cpu,
        **kwargs
    )
    cpu_total = time.time() - cpu_start
    
    # Calculate speedup
    speedup = cpu_timing['total_time'] / gpu_timing['total_time']
    
    # Print detailed comparison
    print("\n" + "="*70)
    print("📊 BENCHMARK RESULTS - CPU vs GPU COMPARISON")
    print("="*70)
    print()
    print(f"{'Step':<30} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>12}")
    print("-"*70)
    print(f"{'1. Frame Extraction':<30} {cpu_timing['step1_time']:>12.2f} {gpu_timing['step1_time']:>12.2f} {cpu_timing['step1_time']/gpu_timing['step1_time']:>11.2f}x")
    print(f"{'2. K-means Clustering':<30} {cpu_timing['step2_time']:>12.2f} {gpu_timing['step2_time']:>12.2f} {cpu_timing['step2_time']/gpu_timing['step2_time']:>11.2f}x")
    print(f"{'3. Full-res Extraction':<30} {cpu_timing['step3_time']:>12.2f} {gpu_timing['step3_time']:>12.2f} {cpu_timing['step3_time']/gpu_timing['step3_time']:>11.2f}x")
    print("-"*70)
    print(f"{'TOTAL':<30} {cpu_timing['total_time']:>12.2f} {gpu_timing['total_time']:>12.2f} {speedup:>11.2f}x")
    print("="*70)
    
    print(f"\n🎯 OVERALL SPEEDUP: {speedup:.2f}x faster with GPU")
    print(f"\n✅ CPU extracted {len(cpu_files)} frames in {cpu_timing['total_time']:.2f}s")
    print(f"✅ GPU extracted {len(gpu_files)} frames in {gpu_timing['total_time']:.2f}s")
    print(f"\n💾 CPU output: {output_dir_cpu}/")
    print(f"💾 GPU output: {output_dir_gpu}/")
    
    return {
        'cpu_timing': cpu_timing,
        'gpu_timing': gpu_timing,
        'speedup': speedup,
        'speedup_step1': cpu_timing['step1_time'] / gpu_timing['step1_time'],
        'speedup_step2': cpu_timing['step2_time'] / gpu_timing['step2_time'],
        'speedup_step3': cpu_timing['step3_time'] / gpu_timing['step3_time'],
        'cpu_files': cpu_files,
        'gpu_files': gpu_files
    }
