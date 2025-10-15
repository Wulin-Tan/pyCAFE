"""Command-line interface for pyCAFE"""

import argparse
import sys
from . import extract_frames_kmeans_gpu, get_video_info, benchmark_cpu_vs_gpu, __version__


def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        prog='pyCAFE',
        description='pyCAFE: Python CUDA Accelerated Frame Extractor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract 50 frames from video
  pyCAFE video.mp4 -o ./frames -n 50
  
  # Extract every 10th frame from middle 80%% of video
  pyCAFE video.mp4 -o ./frames -n 20 --start 0.1 --end 0.9 --step 10
  
  # Show video information
  pyCAFE video.mp4 --info
  
  # Benchmark CPU vs GPU
  pyCAFE video.mp4 --benchmark -n 50
        """
    )
    
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('-o', '--output', default='./output', 
                       help='Output directory (default: ./output)')
    parser.add_argument('-n', '--nframes', type=int, default=10,
                       help='Number of frames to extract (default: 10)')
    parser.add_argument('--start', type=float, default=0.0,
                       help='Start time as fraction 0.0-1.0 (default: 0.0)')
    parser.add_argument('--end', type=float, default=1.0,
                       help='End time as fraction 0.0-1.0 (default: 1.0)')
    parser.add_argument('--step', type=int, default=1,
                       help='Frame sampling step (default: 1)')
    parser.add_argument('--resize', type=int, default=30,
                       help='Clustering resize width (default: 30)')
    parser.add_argument('--color', action='store_true',
                       help='Use color for clustering (default: grayscale)')
    parser.add_argument('--chunk-size', type=int, default=None,
                       help='Manual chunk size (default: auto)')
    parser.add_argument('--info', action='store_true',
                       help='Show video info only')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run CPU vs GPU benchmark')
    parser.add_argument('--version', action='version', 
                       version=f'pyCAFE {__version__}')
    
    args = parser.parse_args()
    
    # Show video info
    if args.info:
        try:
            info = get_video_info(args.video)
            print(f"\n{'='*50}")
            print(f"VIDEO INFORMATION")
            print(f"{'='*50}")
            print(f"Path:       {args.video}")
            print(f"Frames:     {info['nframes']}")
            print(f"FPS:        {info['fps']:.2f}")
            print(f"Duration:   {info['duration']:.2f}s")
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"{'='*50}\n")
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # Run benchmark
    if args.benchmark:
        try:
            results = benchmark_cpu_vs_gpu(
                video_path=args.video,
                output_dir_gpu=args.output + "_gpu",
                output_dir_cpu=args.output + "_cpu",
                n_frames=args.nframes,
                start_time=args.start,
                end_time=args.end,
                step=args.step,
                resize_width=args.resize,
                use_color=args.color,
                chunk_size=args.chunk_size
            )
            return 0
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return 1
    
    # Extract frames
    try:
        frames, timing = extract_frames_kmeans_gpu(
            video_path=args.video,
            output_dir=args.output,
            n_frames=args.nframes,
            start_time=args.start,
            end_time=args.end,
            step=args.step,
            resize_width=args.resize,
            use_color=args.color,
            chunk_size=args.chunk_size
        )
        print(f"\n✅ Successfully extracted {len(frames)} frames to {args.output}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
