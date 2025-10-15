"""Basic usage example for pyCAFE"""

import pyCAFE
import os


def main():
    # Check CUDA availability
    print("="*70)
    print("pyCAFE Basic Usage Example")
    print("="*70)
    
    if pyCAFE.CUDA_AVAILABLE:
        print("✅ CUDA is available - GPU acceleration enabled")
    else:
        print("⚠️  CUDA not available - will use CPU fallback")
    
    # Example video path (replace with your video)
    video_path = "sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"\n❌ Video not found: {video_path}")
        print("Please update the video_path variable with a valid video file")
        return
    
    # Get video info
    print("\n" + "="*70)
    print("STEP 1: Getting Video Information")
    print("="*70)
    
    info = pyCAFE.get_video_info(video_path)
    
    print(f"\nVideo Information:")
    print(f"  Path: {video_path}")
    print(f"  Frames: {info['nframes']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  Resolution: {info['width']}x{info['height']}")
    
    # Extract frames
    print("\n" + "="*70)
    print("STEP 2: Extracting Frames")
    print("="*70)
    
    frames, timing = pyCAFE.extract_frames_kmeans_gpu(
        video_path=video_path,
        output_dir="./output_basic",
        n_frames=20,
        start_time=0.0,
        end_time=1.0,
        step=5,
        resize_width=30,
        use_color=False
    )
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"✅ Extracted {len(frames)} frames")
    print(f"⏱️  Total time: {timing['total_time']:.2f}s")
    print(f"📁 Output directory: ./output_basic")
    print(f"\nTiming breakdown:")
    print(f"  Frame extraction: {timing['step1_time']:.2f}s")
    print(f"  K-means clustering: {timing['step2_time']:.2f}s")
    print(f"  Full-res extraction: {timing['step3_time']:.2f}s")


if __name__ == "__main__":
    main()
