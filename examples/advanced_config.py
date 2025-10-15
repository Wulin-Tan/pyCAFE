"""Advanced configuration example for pyCAFE"""  

import pyCAFE
import os  


def main():  
    print("="*70)  
    print("pyCAFE Advanced Configuration Example")  
    print("="*70)  
    
    # Example video path  
    video_path = "sample_video.mp4"  
    
    if not os.path.exists(video_path):  
        print(f"\n❌ Video not found: {video_path}")  
        print("Please update the video_path variable with a valid video file")  
        return  
    
    # Advanced configuration  
    print("\n" + "="*70)  
    print("CONFIGURATION")  
    print("="*70)  
    
    config = {  
        'video_path': video_path,  
        'output_dir': './output_advanced',  
        'n_frames': 100,              # Extract more frames  
        'start_time': 0.1,            # Skip first 10%  
        'end_time': 0.9,              # Skip last 10%  
        'step': 10,                   # Sample every 10th frame  
        'resize_width': 50,           # Larger thumbnails for clustering  
        'use_color': True,            # Use RGB for clustering (more accurate)
        'chunk_size': 2500,           # Manual chunk size
        'max_iter': 200,              # More K-means iterations
        'tol': 1e-5,                  # Stricter convergence
        'random_state': 123,          # Custom random seed
    }
    
    print("\nExtraction Parameters:")
    for key, value in config.items():
        if key != 'video_path':
            print(f"  {key}: {value}")
    
    # Extract frames with advanced settings
    print("\n" + "="*70)
    print("EXTRACTION")
    print("="*70)
    
    frames, timing = pyCAFE.extract_frames_kmeans_gpu(**config)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"✅ Extracted {len(frames)} frames")
    print(f"⏱️  Total time: {timing['total_time']:.2f}s")
    print(f"📁 Output directory: {config['output_dir']}")
    
    print(f"\nDetailed Timing:")
    print(f"  Frame extraction:   {timing['step1_time']:>8.2f}s ({timing['step1_time']/timing['total_time']*100:>5.1f}%)")
    print(f"  K-means clustering: {timing['step2_time']:>8.2f}s ({timing['step2_time']/timing['total_time']*100:>5.1f}%)")
    print(f"  Full-res extract:   {timing['step3_time']:>8.2f}s ({timing['step3_time']/timing['total_time']*100:>5.1f}%)")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Frames/second: {len(frames)/timing['total_time']:.2f}")
    print(f"  Avg time/frame: {timing['total_time']/len(frames)*1000:.2f}ms")


if __name__ == "__main__":
    main()
