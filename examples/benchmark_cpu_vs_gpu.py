"""Benchmark CPU vs GPU performance"""

import pyCAFE
import os


def main():
    # Example video path (replace with your video)
    video_path = "sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("Please update the video_path variable with a valid video file")
        return
    
    print("="*70)
    print("pyCAFE Benchmark Example - CPU vs GPU")
    print("="*70)
    
    # Run benchmark
    results = pyCAFE.benchmark_cpu_vs_gpu(
        video_path=video_path,
        output_dir_gpu="./benchmark_gpu",
        output_dir_cpu="./benchmark_cpu",
        n_frames=50,
        start_time=0.0,
        end_time=1.0,
        step=5,
        resize_width=30,
        use_color=False
    )
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nOverall Performance:")
    print(f"  CPU Time: {results['cpu_timing']['total_time']:.2f}s")
    print(f"  GPU Time: {results['gpu_timing']['total_time']:.2f}s")
    print(f"  Speedup: {results['speedup']:.2f}x faster with GPU")
    
    print(f"\nPer-Step Speedup:")
    print(f"  Frame Extraction: {results['speedup_step1']:.2f}x")
    print(f"  K-means Clustering: {results['speedup_step2']:.2f}x")
    print(f"  Full-res Extraction: {results['speedup_step3']:.2f}x")
    
    print(f"\nOutput Directories:")
    print(f"  CPU: ./benchmark_cpu/")
    print(f"  GPU: ./benchmark_gpu/")


if __name__ == "__main__":
    main()
