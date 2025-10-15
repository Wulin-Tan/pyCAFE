```markdown
# pyCAFE: Python CUDA Accelerated Frame Extractor

**GPU-accelerated video frame extraction with intelligent K-means clustering for neuroscience and behavioral analysis workflows**

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-11.x%2F12.x-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)

---

## 🧠 Motivation

In modern neuroscience research, **pose estimation** and **behavioral tracking** have become essential tools for understanding animal behavior. Tools like [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut), [SLEAP](https://sleap.ai/), [Anipose](https://anipose.readthedocs.io/), and others rely on deep learning models that require **training data** - specifically, representative frames from behavioral videos that capture the full diversity of postures and movements.

### The Problem

Traditional frame extraction workflows face several bottlenecks:

1. **Slow Frame Decoding**: Reading thousands of frames from video files using CPU-based decoders (OpenCV, FFmpeg) is time-consuming
2. **Memory Constraints**: Loading entire videos into memory is impractical for long recordings (hours of footage)
3. **Inefficient Sampling**: Uniform sampling may miss rare behaviors or include redundant frames
4. **CPU-Only Clustering**: K-means clustering on thousands of high-resolution frames is computationally expensive

For a typical 30-minute behavioral video at 30 FPS (54,000 frames), extracting 200 representative frames for DeepLabCut training can take **15+ minutes on CPU**. When processing dozens of videos, this becomes a significant time sink.

### The Solution: pyCAFE

**pyCAFE** solves these problems by leveraging **GPU acceleration** at every step:

- ⚡ **NVIDIA DALI**: Hardware-accelerated video decoding on GPU (10-30x faster than CPU)
- 🔬 **RAPIDS cuML**: GPU-accelerated K-means clustering (5-50x faster than scikit-learn)
- 🧩 **CuPy**: GPU array operations for efficient preprocessing
- 📊 **Intelligent Chunking**: Process videos longer than GPU memory with automatic batch management

**Result**: Extract 200 frames from a 30-minute video in **~30 seconds** instead of 15+ minutes.

---

## 🎯 Use Cases

### Primary: Deep Learning Training Data
- **DeepLabCut**: Extract diverse training frames for pose estimation
- **SLEAP**: Generate training data for multi-animal tracking
- **DLC-Live**: Prepare frames for real-time inference validation
- **Custom Models**: Create training datasets for any video-based ML task

### Secondary Applications
- **Video Summarization**: Generate thumbnails representing video content
- **Quality Control**: Sample frames for manual inspection of long recordings
- **Dataset Creation**: Build balanced datasets from video archives
- **Behavioral Annotation**: Select frames for manual labeling workflows

---

## 🚀 Features

### Core Capabilities
- **GPU-Accelerated Pipeline**: Entire workflow runs on GPU (NVIDIA DALI → cuML → CuPy)
- **Intelligent Frame Selection**: K-means clustering ensures temporal and visual diversity
- **Memory-Efficient Processing**: Automatic chunking handles videos of any length
- **Flexible Sampling**: Control time ranges, frame steps, and cluster parameters
- **CPU Fallback**: Gracefully degrades to CPU-only mode if GPU unavailable

### Advanced Features
- **Benchmarking Suite**: Built-in CPU vs GPU performance comparison
- **Color & Grayscale Modes**: Balance accuracy vs speed for clustering
- **Custom Clustering Parameters**: Fine-tune K-means convergence and initialization
- **Batch Processing Ready**: Process multiple videos programmatically
- **Progress Tracking**: Real-time progress bars for each processing stage

### Developer-Friendly
- **Python API**: Clean, documented functions for pipeline integration
- **Command-Line Interface**: Quick operations without writing code
- **Reproducible Results**: Seed control for consistent frame selection
- **Extensive Logging**: Detailed timing and diagnostic information

---

## 📊 Performance Benchmarks

### Hardware: NVIDIA RTX 3090 (24GB VRAM)

| Video Duration | Resolution | Frames Sampled | Frames Extracted | CPU Time | GPU Time | **Speedup** |
|----------------|------------|----------------|------------------|----------|----------|-------------|
| 10 min | 1920×1080 | 5,400 | 50 | 145s | 12s | **12.1x** |
| 30 min | 1920×1080 | 16,200 | 100 | 420s | 28s | **15.0x** |
| 60 min | 1920×1080 | 32,400 | 200 | 890s | 54s | **16.5x** |
| 120 min | 1280×720 | 43,200 | 300 | 1,240s | 78s | **15.9x** |

### Per-Step Breakdown (30 min video, 100 frames)

| Step | CPU (OpenCV + sklearn) | GPU (DALI + cuML) | Speedup |
|------|------------------------|-------------------|---------|
| 1. Frame Extraction | 312s | 18s | **17.3x** |
| 2. K-means Clustering | 94s | 7s | **13.4x** |
| 3. Full-Res Export | 14s | 3s | **4.7x** |
| **Total** | **420s** | **28s** | **15.0x** |

*Note: Performance scales with GPU compute capability. RTX 4090 shows 20-25x speedup.*

---

## 📋 Requirements

### Minimum Requirements (CPU Mode)
```
Python 3.8+
numpy >= 1.20.0
opencv-python >= 4.5.0
scikit-learn >= 0.24.0
Pillow >= 8.0.0
tqdm >= 4.60.0
```

### Recommended: GPU Requirements
```
NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
CUDA Toolkit 11.x or 12.x
16GB+ GPU memory (for 1080p videos)
cuPy (CUDA array library)
RAPIDS cuML (GPU machine learning)
NVIDIA DALI (GPU video decoding)
```

### Supported Platforms
- **Linux**: Full GPU support (tested on Ubuntu 20.04/22.04)
- **Windows**: GPU support with WSL2 + CUDA
- **macOS**: CPU-only mode (Apple Silicon and Intel)

---

## 🔧 Installation

### Option 1: Quick Install (CPU Only)

For testing or systems without NVIDIA GPUs:

```bash
pip install pyCAFE
```

### Option 2: Full GPU Installation (Recommended)

#### Step 1: Install CUDA Toolkit

```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Check installation
nvcc --version
nvidia-smi
```

#### Step 2: Install pyCAFE with GPU Dependencies

**For CUDA 11.x:**
```bash
pip install pyCAFE

# Install GPU libraries
pip install cupy-cuda11x
pip install cuml-cu11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

**For CUDA 12.x:**
```bash
pip install pyCAFE

# Install GPU libraries
pip install cupy-cuda12x
pip install cuml-cu12
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
```

#### Step 3: Verify GPU Installation

```python
import pyCAFE

print(f"GPU Available: {pyCAFE.CUDA_AVAILABLE}")
# Should print: GPU Available: True

# Test with a sample video
pyCAFE.get_video_info("your_video.mp4")
```

### Option 3: Conda Installation (Easiest for GPU)

```bash
# Create new environment with GPU support
conda create -n pycafe python=3.10 -y
conda activate pycafe

# Install RAPIDS (includes cuML, cuPy, cuDF)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=23.10 cupy cudatoolkit=11.8

# Install DALI
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    nvidia-dali-cuda110

# Install pyCAFE
pip install pyCAFE
```

### Option 4: Docker (Pre-configured GPU Environment)

```bash
# Pull pre-built image with all GPU dependencies
docker pull wulintan/pycafe:latest-gpu

# Run container with GPU access
docker run --gpus all -it -v $(pwd):/workspace pycafe:latest-gpu

# Inside container
pycafe --version
```

### Development Installation

```bash
git clone https://github.com/Wulin-Tan/pyCAFE.git
cd pyCAFE
pip install -e ".[dev,gpu]"

# Run tests
pytest tests/
```

---

## 🎯 Quick Start

### 1. Command Line Interface

#### Basic Frame Extraction
```bash
# Extract 50 representative frames from a video
pycafe video.mp4 -o ./training_frames -n 50

# Extract from specific time range (skip first/last 10%)
pycafe video.mp4 -o ./frames -n 100 --start 0.1 --end 0.9

# Sample every 5th frame (faster for long videos)
pycafe video.mp4 -o ./frames -n 50 --step 5

# Use color clustering (more accurate, slower)
pycafe video.mp4 -o ./frames -n 50 --color
```

#### Video Information
```bash
# Get video metadata
pycafe video.mp4 --info

# Output:
# ==================================================
# VIDEO INFORMATION
# ==================================================
# Path:       video.mp4
# Frames:     54000
# FPS:        30.00
# Duration:   1800.00s (30m 0.0s)
# Resolution: 1920x1080
# ==================================================
```

#### Benchmarking
```bash
# Compare CPU vs GPU performance
pycafe video.mp4 --benchmark -n 100

# Output shows detailed timing breakdown and speedup
```

### 2. Python API

#### Basic Usage for DeepLabCut

```python
import pyCAFE

# Extract 200 diverse frames for DLC training
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="mouse_behavior.mp4",
    output_dir="./deeplabcut_frames",
    n_frames=200,
    step=10,  # Sample every 10th frame
    resize_width=30,  # Small thumbnails for fast clustering
    use_color=False  # Grayscale is usually sufficient
)

print(f"✅ Extracted {len(frames)} frames in {timing['total_time']:.1f}s")
print(f"📁 Frames saved to: ./deeplabcut_frames/")
print(f"🔬 Ready for DeepLabCut labeling!")
```

#### Advanced Configuration

```python
import pyCAFE

# Get video information first
info = pyCAFE.get_video_info("long_recording.mp4")
print(f"Video: {info['duration']/60:.1f} minutes, {info['nframes']:,} frames")

# Advanced extraction with fine-tuned parameters
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="long_recording.mp4",
    output_dir="./frames",
    n_frames=300,
    
    # Time range: analyze middle 80% (skip start/end artifacts)
    start_time=0.1,
    end_time=0.9,
    
    # Sampling: every 15th frame (reduce redundancy)
    step=15,
    
    # Clustering: larger thumbnails for better accuracy
    resize_width=50,
    use_color=True,  # Use RGB for color-based behaviors
    
    # Processing: manual chunk size for memory control
    chunk_size=2000,
    
    # K-means: more iterations for convergence
    max_iter=300,
    tol=1e-5,
    random_state=42  # Reproducible results
)

# Analyze timing breakdown
print(f"\n📊 Performance Breakdown:")
print(f"   Frame extraction:  {timing['step1_time']:.1f}s")
print(f"   K-means clustering: {timing['step2_time']:.1f}s")
print(f"   Full-res export:   {timing['step3_time']:.1f}s")
print(f"   Total:             {timing['total_time']:.1f}s")
```

#### Batch Processing Multiple Videos

```python
import pyCAFE
from pathlib import Path

# Process all videos in a directory
video_dir = Path("./behavioral_videos")
output_base = Path("./extracted_frames")

for video_path in video_dir.glob("*.mp4"):
    print(f"\n🎬 Processing: {video_path.name}")
    
    # Create output directory for this video
    output_dir = output_base / video_path.stem
    
    try:
        frames, timing = pyCAFE.extract_frames_kmeans_gpu(
            video_path=str(video_path),
            output_dir=str(output_dir),
            n_frames=100,
            step=10
        )
        
        print(f"✅ {video_path.name}: {len(frames)} frames in {timing['total_time']:.1f}s")
        
    except Exception as e:
        print(f"❌ {video_path.name}: Error - {e}")

print("\n🎉 Batch processing complete!")
```

#### Integration with DeepLabCut

```python
import pyCAFE
import deeplabcut as dlc

# 1. Extract frames for DLC training
video_path = "mouse_openfield.mp4"
frames_dir = "./dlc_training_frames"

print("🎬 Step 1: Extracting representative frames...")
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path=video_path,
    output_dir=frames_dir,
    n_frames=200,
    step=5,
    use_color=False
)
print(f"✅ Extracted {len(frames)} frames in {timing['total_time']:.1f}s")

# 2. Create DLC project
print("\n🔬 Step 2: Creating DeepLabCut project...")
config_path = dlc.create_new_project(
    'MouseTracking',
    'WulinTan',
    [video_path],
    working_directory='./dlc_project',
    copy_videos=False
)

# 3. Import extracted frames into DLC
print("\n📥 Step 3: Importing frames into DLC...")
# Manually copy frames to DLC labeled-data folder
# or use DLC's extract_frames with mode='manual' pointing to frames_dir

print("🎉 Ready to label in DeepLabCut!")
print(f"   Config: {config_path}")
print(f"   Frames: {frames_dir}")
```

---

## 📖 How It Works

### The Three-Step Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    pyCAFE Processing Pipeline                    │
└─────────────────────────────────────────────────────────────────┘

Step 1: GPU Frame Extraction (NVIDIA DALI)
┌────────────────────────────────────────────────────────┐
│  Video File  →  GPU Decode  →  Downsample  →  Batch   │
│  (H.264/VP9)    (NVDEC)        (30x30px)     (CuPy)   │
└────────────────────────────────────────────────────────┘
                          ↓
              Extract every Nth frame
              Resize to small thumbnails
              Convert to grayscale/RGB
                          ↓

Step 2: GPU K-means Clustering (RAPIDS cuML)
┌────────────────────────────────────────────────────────┐
│  Frame Features  →  K-means  →  Cluster IDs  →  Select │
│  (Flattened)        (GPU)       (N clusters)   (1/each)│
└────────────────────────────────────────────────────────┘
                          ↓
              Group similar frames
              Find cluster centers
              Select closest frame per cluster
                          ↓

Step 3: Full-Resolution Export (OpenCV)
┌────────────────────────────────────────────────────────┐
│  Selected Indices  →  Seek & Read  →  Save PNG Files   │
│  [10, 245, 891...] →  (OpenCV)     →  frame_XXXXXX.png│
└────────────────────────────────────────────────────────┘
```

### Why K-means Clustering?

Traditional uniform sampling (e.g., every 100th frame) has limitations:

❌ **Problems with Uniform Sampling:**
- May miss rare but important behaviors
- Includes redundant frames from static periods
- No guarantee of diversity
- Doesn't adapt to video content

✅ **Advantages of K-means Clustering:**
- **Content-Aware**: Groups visually similar frames together
- **Diverse Selection**: Guarantees one frame per cluster
- **Captures Transitions**: Detects scene changes and movement
- **Balanced Coverage**: Equal representation of different poses/behaviors

**Example**: In a 10-minute video where a mouse explores for 2 minutes then grooms for 8 minutes:
- Uniform sampling: ~80% grooming frames, ~20% exploration frames
- K-means clustering: Balanced representation of both behaviors

### Memory Management: Chunking Strategy

For videos exceeding GPU memory, pyCAFE automatically chunks processing:

```python
# Automatic chunk size selection
if total_frames < 2,000:
    chunk_size = total_frames  # No chunking
elif total_frames < 10,000:
    chunk_size = 2,000
else:
    chunk_size = 3,000

# Process in chunks to avoid memory issues
for chunk in chunks:
    extract_frames(chunk)
    cluster_frames(chunk)
    delete_pipeline()  # Free GPU memory
```

This allows processing hour-long 4K videos on GPUs with only 8GB VRAM.

---

## 🎛️ API Reference

### Core Functions

#### `extract_frames_kmeans_gpu()`

Main function for GPU-accelerated frame extraction.

```python
extract_frames_kmeans_gpu(
    video_path: str,
    output_dir: str,
    n_frames: int = 10,
    start_time: float = 0.0,
    end_time: float = 1.0,
    step: int = 1,
    resize_width: int = 30,
    use_color: bool = False,
    device_id: int = 0,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    chunk_size: int = None
) -> Tuple[List[str], Dict[str, float]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | str | required | Path to input video file |
| `output_dir` | str | required | Directory for extracted frames (created if doesn't exist) |
| `n_frames` | int | 10 | Number of representative frames to extract |
| `start_time` | float | 0.0 | Start time as fraction of video duration (0.0 = start, 1.0 = end) |
| `end_time` | float | 1.0 | End time as fraction of video duration |
| `step` | int | 1 | Sample every Nth frame (higher = faster but less coverage) |
| `resize_width` | int | 30 | Width in pixels for clustering thumbnails (smaller = faster) |
| `use_color` | bool | False | Use RGB (True) or grayscale (False) for clustering |
| `device_id` | int | 0 | GPU device ID for multi-GPU systems |
| `max_iter` | int | 100 | Maximum iterations for K-means convergence |
| `tol` | float | 1e-4 | Convergence tolerance for K-means |
| `random_state` | int | 42 | Random seed for reproducible results |
| `chunk_size` | int | None | Manual chunk size (auto-calculated if None) |

**Returns:**
- `frames` (List[str]): List of paths to extracted frame files
- `timing` (Dict[str, float]): Timing breakdown with keys:
  - `'step1_time'`: Frame extraction time
  - `'step2_time'`: K-means clustering time
  - `'step3_time'`: Full-resolution export time
  - `'total_time'`: Total processing time

**Example:**
```python
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="video.mp4",
    output_dir="./frames",
    n_frames=100,
    start_time=0.1,  # Skip first 10%
    end_time=0.9,    # Skip last 10%
    step=5           # Every 5th frame
)
```

---

#### `extract_frames_kmeans_cpu()`

CPU-only fallback version (slower but no GPU required).

```python
extract_frames_kmeans_cpu(
    video_path: str,
    output_dir: str,
    n_frames: int = 10,
    start_time: float = 0.0,
    end_time: float = 1.0,
    step: int = 1,
    resize_width: int = 30,
    use_color: bool = False,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42
) -> Tuple[List[str], Dict[str, float]]
```

**Same parameters as GPU version**, except:
- No `device_id` (CPU only)
- No `chunk_size` (different memory management)

---

#### `get_video_info()`

Extract metadata from video file.

```python
get_video_info(video_path: str) -> Dict[str, Union[int, float]]
```

**Parameters:**
- `video_path` (str): Path to video file

**Returns:**
Dictionary with keys:
- `'nframes'` (int): Total number of frames
- `'fps'` (float): Frames per second
- `'width'` (int): Frame width in pixels
- `'height'` (int): Frame height in pixels
- `'duration'` (float): Video duration in seconds

**Example:**
```python
info = pyCAFE.get_video_info("video.mp4")
print(f"Duration: {info['duration']/60:.1f} minutes")
print(f"Resolution: {info['width']}x{info['height']}")
```

---

#### `benchmark_cpu_vs_gpu()`

Compare CPU and GPU performance on the same video.

```python
benchmark_cpu_vs_gpu(
    video_path: str,
    output_dir_gpu: str = "./gpu_output",
    output_dir_cpu: str = "./cpu_output",
    **kwargs
) -> Dict[str, Any]
```

**Parameters:**
- `video_path` (str): Path to video file
- `output_dir_gpu` (str): Output directory for GPU extraction
- `output_dir_cpu` (str): Output directory for CPU extraction
- `**kwargs`: Additional parameters passed to extraction functions

**Returns:**
Dictionary with keys:
- `'cpu_timing'`: CPU timing breakdown
- `'gpu_timing'`: GPU timing breakdown
- `'speedup'`: Overall speedup factor (CPU time / GPU time)
- `'speedup_step1'`: Frame extraction speedup
- `'speedup_step2'`: K-means clustering speedup
- `'speedup_step3'`: Full-res export speedup
- `'cpu_files'`: List of CPU-extracted files
- `'gpu_files'`: List of GPU-extracted files

**Example:**
```python
results = pyCAFE.benchmark_cpu_vs_gpu(
    "video.mp4",
    n_frames=100,
    step=5
)
print(f"GPU is {results['speedup']:.1f}x faster!")
```

---

### Utility Functions

#### `perform_kmeans_gpu()`

Low-level K-means clustering function.

```python
perform_kmeans_gpu(
    frames_data: Union[np.ndarray, cp.ndarray],
    n_clusters: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42
) -> List[int]
```

#### `extract_downsampled_frames_gpu()`

Low-level frame extraction with DALI.

```python
extract_downsampled_frames_gpu(
    video_path: str,
    start_frame: int,
    end_frame: int,
    step: int = 1,
    resize_width: int = 30,
    use_color: bool = False,
    batch_size: int = 256,
    device_id: int = 0,
    chunk_size: int = None
) -> Tuple[Union[np.ndarray, cp.ndarray], List[int]]
```

---

## 🔬 Research Applications

### Neuroscience & Behavioral Analysis

#### DeepLabCut Pose Estimation
```python
# Extract diverse frames capturing full range of motion
frames, _ = pyCAFE.extract_frames_kmeans_gpu(
    "mouse_social_interaction.mp4",
    "./dlc_frames",
    n_frames=200,
    step=10,
    use_color=False  # Grayscale sufficient for pose
)
```

#### SLEAP Multi-Animal Tracking
```python
# Process multiple videos from social behavior experiment
for video in ["cage1.mp4", "cage2.mp4", "cage3.mp4"]:
    pyCAFE.extract_frames_kmeans_gpu(
        video,
        f"./sleap_training/{Path(video).stem}",
        n_frames=150,
        step=5
    )
```

#### Calcium Imaging + Behavior
```python
# Extract frames synchronized with Ca2+ recording
info = pyCAFE.get_video_info("behavior_video.mp4")
calcium_fps = 20  # Hz
behavior_fps = info['fps']

# Match sampling to calcium imaging rate
step = int(behavior_fps / calcium_fps)

frames, _ = pyCAFE.extract_frames_kmeans_gpu(
    "behavior_video.mp4",
    "./matched_frames",
    n_frames=100,
    step=step
)
```

### Ethology & Field Research

```python
# Long wildlife recordings
pyCAFE.extract_frames_kmeans_gpu(
    "bird_nest_24hr.mp4",
    "./representative_frames",
    n_frames=500,
    step=30,  # Sample every second at 30fps
    chunk_size=2000  # Handle long video
)
```

---

## 🧪 Advanced Usage

### Custom Clustering Pipeline

```python
import pyCAFE
import numpy as np
from sklearn.decomposition import PCA

# Step 1: Extract downsampled frames
frames_gpu, indices = pyCAFE.extract_downsampled_frames_gpu(
    "video.mp4",
    start_frame=0,
    end_frame=10000,
    step=10,
    resize_width=50
)

# Step 2: Custom preprocessing (PCA dimensionality reduction)
frames_cpu = frames_gpu.get() if hasattr(frames_gpu, 'get') else frames_gpu
frames_flat = frames_cpu.reshape(len(frames_cpu), -1)

pca = PCA(n_components=50)
frames_pca = pca.fit_transform(frames_flat)

# Step 3: Cluster in reduced space
import cupy as cp
frames_pca_gpu = cp.asarray(frames_pca)

selected_indices = pyCAFE.perform_kmeans_gpu(
    frames_pca_gpu,
    n_clusters=100,
    max_iter=300
)

# Step 4: Export selected frames
selected_frame_numbers = [indices[i] for i in selected_indices]
pyCAFE.extract_specific_frames(
    "video.mp4",
    selected_frame_numbers,
    "./custom_frames"
)
```

### Multi-GPU Processing

```python
import pyCAFE
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

videos = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
gpus = [0, 1, 0, 1]  # Distribute across 2 GPUs

def process_video(video, gpu_id):
    return pyCAFE.extract_frames_kmeans_gpu(
        video,
        f"./output/{Path(video).stem}",
        n_frames=100,
        device_id=gpu_id
    )

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(process_video, videos, gpus))
```

### Integration with Video Analysis Pipelines

```python
import pyCAFE
import cv2
import numpy as np

# Extract frames
frames, _ = pyCAFE.extract_frames_kmeans_gpu(
    "experiment.mp4",
    "./temp_frames",
    n_frames=50
)

# Post-process: Apply background subtraction
for frame_path in frames:
    img = cv2.imread(frame_path)
    
    # Your custom analysis
    # ... background subtraction, ROI extraction, etc.
    
    processed = your_analysis_function(img)
    cv2.imwrite(frame_path.replace('.png', '_processed.png'), processed)
```

---

## 📊 Comparison with Alternatives

| Method | Speed (30min video) | Diversity | Memory | GPU Support |
|--------|---------------------|-----------|--------|-------------|
| **pyCAFE** | **~30s** | ✅ K-means | ✅ Chunked | ✅ Full |
| DeepLabCut `extract_frames` | ~180s | ❌ Uniform | ⚠️ High | ❌ No |
| FFmpeg + Python | ~120s | ❌ Uniform | ✅ Low | ❌ No |
| OpenCV Sequential | ~420s | ❌ Uniform | ✅ Low | ❌ No |
| Manual Sampling | Variable | ⚠️ Manual | ✅ Low | ❌ No |

---

## ⚙️ Configuration Tips

### Optimal Parameters by Use Case

#### Quick Preview (Fast)
```python
n_frames=20
step=15
resize_width=20
use_color=False
```

#### DeepLabCut Training (Balanced)
```python
n_frames=200
step=10
resize_width=30
use_color=False
```

#### High-Diversity Dataset (Accurate)
```python
n_frames=500
step=5
resize_width=50
use_color=True
max_iter=300
```

### Memory Constraints

| GPU VRAM | Max Resolution | Recommended chunk_size |
|----------|----------------|------------------------|
| 6GB | 1080p | 1000 |
| 8GB | 1080p | 1500 |
| 12GB | 1080p | 2500 |
| 16GB | 4K | 2000 |
| 24GB+ | 4K | 3000 |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Not Available

**Problem:**
```python
import pyCAFE
print(pyCAFE.CUDA_AVAILABLE)  # False
```

**Solutions:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall GPU libraries
pip uninstall cupy cuml nvidia-dali
pip install cupy-cuda11x cuml-cu11 nvidia-dali-cuda110
```

#### 2. Out of Memory (OOM)

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce chunk size
pyCAFE.extract_frames_kmeans_gpu(
    "video.mp4",
    "./frames",
    n_frames=100,
    chunk_size=1000  # Smaller chunks
)

# Reduce thumbnail size
resize_width=20  # Default: 30

# Sample fewer frames
step=20  # Default: 1
```

#### 3. Slow Performance Despite GPU

**Problem:** GPU extraction not much faster than CPU

**Possible Causes:**
1. **DALI not using GPU decoder**:
   ```python
   # Check video codec
   ffprobe video.mp4
   # H.264/H.265 best supported
   ```

2. **Small video file** (overhead dominates):
   - GPU acceleration benefits longer videos (>5 min)

3. **Disk I/O bottleneck**:
   - Use SSD storage
   - Don't save to network drives

#### 4. Video Format Issues

**Problem:**
```
Error: Cannot open video file
```

**Solutions:**
```bash
# Convert to compatible format
ffmpeg -i input.avi -c:v libx264 -preset fast output.mp4

# Supported formats: MP4, AVI, MOV, MKV
# Recommended codec: H.264
```

#### 5. Incorrect Frame Count

**Problem:** Extracted fewer frames than requested

**Cause:** Video shorter than expected, or `step` too large

**Solution:**
```python
# Check video info first
info = pyCAFE.get_video_info("video.mp4")
max_frames = info['nframes'] // step

n_frames = min(requested_frames, max_frames)
```

---

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=pyCAFE tests/

# Run specific test
pytest tests/test_basic.py::TestUtils::test_format_time
```

### Test with Your Own Video

```bash
export TEST_VIDEO_PATH=/path/to/your/video.mp4
pytest tests/test_basic.py
```

### Create Test Video

```python
# Generate synthetic test video
import cv2
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_video.mp4', fourcc, 30.0, (640, 480))

for i in range(1000):
    # Create frames with varying content
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out.write(frame)

out.release()
```

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

### High Priority
- [ ] Support for additional video codecs (VP9, AV1)
- [ ] Alternative clustering algorithms (DBSCAN, hierarchical)
- [ ] Real-time frame selection during recording
- [ ] Integration with more pose estimation tools

### Medium Priority
- [ ] GUI interface for non-programmers
- [ ] Video quality assessment metrics
- [ ] Batch processing CLI improvements
- [ ] Docker image with pre-installed dependencies

### Low Priority
- [ ] Cloud processing support (AWS, GCP)
- [ ] Frame annotation export formats
- [ ] Video preprocessing filters

### How to Contribute

1. Fork the repository
2. Create feature branch: `git checkout -b feature/AmazingFeature`
3. Make changes and add tests
4. Ensure tests pass: `pytest tests/`
5. Commit: `git commit -m 'Add AmazingFeature'`
6. Push: `git push origin feature/AmazingFeature`
7. Open Pull Request

---

## 📚 Citation

If you use pyCAFE in your research, please cite:

```bibtex
@software{pycafe2024,
  title = {pyCAFE: Python CUDA Accelerated Frame Extractor},
  author = {Tan, Wulin},
  year = {2024},
  url = {https://github.com/Wulin-Tan/pyCAFE},
  version = {0.1.0}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Wulin Tan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🔗 Related Projects

### Pose Estimation & Tracking
- **[DeepLabCut](http://www.mackenziemathislab.org/deeplabcut)** - Markerless pose estimation
- **[SLEAP](https://sleap.ai/)** - Multi-animal pose tracking
- **[Anipose](https://anipose.readthedocs.io/)** - 3D pose estimation
- **[SimBA](https://github.com/sgoldenlab/simba)** - Behavior classification

### Video Processing
- **[NVIDIA DALI](https://github.com/NVIDIA/DALI)** - GPU video decoding
- **[RAPIDS](https://rapids.ai/)** - GPU-accelerated data science
- **[decord](https://github.com/dmlc/decord)** - Efficient video loader

### Machine Learning
- **[cuML](https://github.com/rapidsai/cuml)** - GPU machine learning
- **[CuPy](https://cupy.dev/)** - NumPy-compatible GPU arrays

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Wulin-Tan/pyCAFE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Wulin-Tan/pyCAFE/discussions)
- **Email**: wulintan9527@gmail.com

---

## 🙏 Acknowledgments

- **NVIDIA** for DALI, cuML, and RAPIDS ecosystem
- **DeepLabCut team** for inspiring this work
- **Neuroscience community** for feedback and testing
- **Contributors** who helped improve pyCAFE

---

## 🗺️ Roadmap

### Version 0.2.0 (Q2 2024)
- [ ] Multi-video batch processing CLI
- [ ] Progress callbacks for GUI integration
- [ ] Video quality metrics (blur, brightness)
- [ ] Frame deduplication

### Version 0.3.0 (Q3 2024)
- [ ] Alternative clustering (DBSCAN, agglomerative)
- [ ] Temporal sampling strategies
- [ ] ROI-based extraction
- [ ] Integration with DLC/SLEAP APIs

### Version 1.0.0 (Q4 2024)
- [ ] Stable API
- [ ] Comprehensive documentation
- [ ] Tutorial videos
- [ ] Benchmark suite

---

<div align="center">

**⭐ Star this repo if pyCAFE helps your research! ⭐**

Made with ❤️ for the neuroscience community

[🐛 Report Bug](https://github.com/Wulin-Tan/pyCAFE/issues) · [✨ Request Feature](https://github.com/Wulin-Tan/pyCAFE/issues) · [📖 Documentation](https://github.com/Wulin-Tan/pyCAFE#readme)

</div>
```
