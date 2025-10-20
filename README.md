# pyCAFE: Python CUDA Accelerated Frame Extractor

**GPU-accelerated video frame extraction with intelligent K-means clustering for neuroscience and behavioral analysis workflows**

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-11.x%2F12.x-brightgreen)
![Status](https://img.shields.io/badge/status-active-success)

---

## 🧠 Overview

Modern neuroscience research requires extracting representative training frames from behavioral videos for pose estimation and tracking analysis. **pyCAFE** leverages GPU acceleration to dramatically speed up this process while ensuring visual and temporal diversity through intelligent clustering.

### Key Features

- ⚡ **GPU-Accelerated Pipeline**: NVIDIA DALI for video decoding, RAPIDS cuML for clustering
- 🔬 **Intelligent Frame Selection**: K-means clustering ensures temporal and visual diversity
- 📊 **Memory-Efficient**: Automatic chunking handles videos of any length
- 🎯 **Research-Ready**: Optimized for DeepLabCut, SLEAP, and behavioral analysis workflows
- 💻 **CPU Fallback**: Gracefully degrades when GPU unavailable

---

## 📊 Performance Benchmarks

### Overall Performance Comparison

| Video Duration | Resolution | Total Frames | Source | Frames Extracted | CPU Time | GPU Time | **Speedup** |
|----------------|------------|--------------|--------|------------------|----------|----------|-------------|
| **1 min** | 1280×720 | 1,981 | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.mw6m905v3) | 50 | 29.5s | 2.0s | **14.7x** |
| **15 min** | 640×480 | 27,000 | [Zenodo](https://zenodo.org/records/4629544) | 50 | 66.7s | 9.5s | **7.0x** |
| **55 min** | 480×480 | 99,150 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FSAPNJG) | 50 | 144.5s | 36.7s | **3.9x** |

### Detailed Step-by-Step Analysis

#### Test 1: Short Video (1 min)
**File**: `2018-4-30_NP_A11_F_N_S_25.mp4` | **Resolution**: 1280×720 | **Total Frames**: 1,981

| Processing Step | CPU Time | GPU Time | **Speedup** |
|-----------------|----------|----------|-------------|
| Frame Extraction | 24.3s | 1.1s | **23.2x** |
| K-means Clustering | 0.4s | 0.1s | **4.1x** |
| Full-res Export | 4.8s | 0.9s | **5.6x** |
| **Total Pipeline** | **29.5s** | **2.0s** | **14.7x** |

#### Test 2: Medium Video (15 min)
**File**: `95-WT-male-56-20200615.avi` | **Resolution**: 640×480 | **Total Frames**: 27,000

| Processing Step | CPU Time | GPU Time | **Speedup** |
|-----------------|----------|----------|-------------|
| Frame Extraction | 60.3s | 8.3s | **7.2x** |
| K-means Clustering | 5.4s | 0.1s | **56.9x** |
| Full-res Export | 1.1s | 1.1s | **1.0x** |
| **Total Pipeline** | **66.7s** | **9.5s** | **7.0x** |

#### Test 3: Long Video (55 min)
**File**: `LL1-1_BalbcJ.mp4` | **Resolution**: 480×480 | **Total Frames**: 99,150

| Processing Step | CPU Time | GPU Time | **Speedup** |
|-----------------|----------|----------|-------------|
| Frame Extraction | 119.0s | 35.9s | **3.3x** |
| K-means Clustering | 24.9s | 0.2s | **134.3x** |
| Full-res Export | 0.7s | 0.7s | **1.0x** |
| **Total Pipeline** | **144.5s** | **36.7s** | **3.9x** |

### Key Performance Insights

- **K-means Clustering**: Achieves up to **134x speedup** on GPU, making it nearly instantaneous even for large datasets
- **Frame Extraction**: **3-23x faster** with GPU hardware decoding, with greater speedup on high-resolution videos
- **Scaling**: GPU acceleration provides consistent benefits across video lengths, from 1-minute clips to hour-long recordings
- **Memory Efficiency**: Automatic chunking enables processing of videos exceeding GPU memory capacity

*Benchmarked on NVIDIA Tesla T4 (16GB) with Intel Xeon (Cascadelake) 8 vCPU @ 56GB RAM*

---

## 📋 Requirements

### Minimum (CPU Mode)
- Python 3.8+
- numpy, opencv-python, scikit-learn, Pillow, tqdm

### GPU Acceleration (Recommended)
- NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
- CUDA Toolkit 11.x or 12.x
- cuPy, RAPIDS cuML, NVIDIA DALI

**Supported Platforms**: Linux (full GPU support), Windows (WSL2 + CUDA), macOS (CPU-only)

---

## 🔧 Installation

### Quick Install (CPU Only)
```bash
pip install pyCAFE
```

### GPU Installation (Recommended)

**CUDA 11.x:**
```bash
pip install pyCAFE
pip install cupy-cuda11x cuml-cu11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

**CUDA 12.x:**
```bash
pip install pyCAFE
pip install cupy-cuda12x cuml-cu12
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
```

### Conda (Easiest for GPU)
```bash
conda create -n pycafe python=3.10 -y
conda activate pycafe
conda install -c rapidsai -c conda-forge -c nvidia cuml=23.10 cupy cudatoolkit=11.8
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
pip install pyCAFE
```

### Verify Installation
```python
import pyCAFE
print(f"GPU Available: {pyCAFE.CUDA_AVAILABLE}")
```

---

## 🎯 Quick Start

### Command Line

```bash
# Extract 50 representative frames
pycafe video.mp4 -o ./frames -n 50

# Extract from specific time range
pycafe video.mp4 -o ./frames -n 100 --start 0.1 --end 0.9

# Sample every 5th frame for faster processing
pycafe video.mp4 -o ./frames -n 50 --step 5

# Get video information
pycafe video.mp4 --info

# Benchmark CPU vs GPU performance
pycafe video.mp4 --benchmark -n 50 --step 5
```

### Python API

#### Basic Usage
```python
import pyCAFE

# Extract diverse frames
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="behavior.mp4",
    output_dir="./frames",
    n_frames=200,
    step=10
)

print(f"Extracted {len(frames)} frames in {timing['total_time']:.1f}s")
```

#### Benchmark Your Video
```python
import pyCAFE

# Compare CPU vs GPU performance
results = pyCAFE.benchmark_cpu_vs_gpu(
    video_path="your_video.mp4",
    n_frames=50,
    step=5
)

print(f"GPU is {results['speedup']:.2f}x faster")
print(f"Frame extraction speedup: {results['speedup_step1']:.2f}x")
print(f"K-means clustering speedup: {results['speedup_step2']:.2f}x")
```

#### Advanced Configuration
```python
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="recording.mp4",
    output_dir="./frames",
    n_frames=300,
    start_time=0.1,      # Skip first 10%
    end_time=0.9,        # Skip last 10%
    step=15,             # Every 15th frame
    resize_width=50,     # Larger thumbnails
    use_color=True,      # RGB clustering
    max_iter=300,        # More iterations
    random_state=42      # Reproducible
)
```

#### Batch Processing
```python
from pathlib import Path

video_dir = Path("./videos")
output_base = Path("./extracted_frames")

for video_path in video_dir.glob("*.mp4"):
    output_dir = output_base / video_path.stem
    
    frames, timing = pyCAFE.extract_frames_kmeans_gpu(
        video_path=str(video_path),
        output_dir=str(output_dir),
        n_frames=100,
        step=5
    )
    print(f"✅ {video_path.name}: {len(frames)} frames in {timing['total_time']:.1f}s")
```

---

## 📖 How It Works

### Three-Step Pipeline

1. **GPU Frame Extraction (NVIDIA DALI)**
   - Hardware-accelerated video decoding
   - Downsample to small thumbnails for clustering
   - Process in chunks to manage memory

2. **GPU K-means Clustering (RAPIDS cuML)**
   - Group visually similar frames
   - Select one representative frame per cluster
   - Ensures diversity across time and visual content

3. **Full-Resolution Export (OpenCV)**
   - Extract selected frames at original quality
   - Save as PNG files with frame numbers

### Why K-means Clustering?

**Traditional uniform sampling** (every Nth frame) may miss rare behaviors or include redundant frames.

**K-means clustering** provides:
- Content-aware selection
- Guaranteed diversity (one frame per cluster)
- Balanced representation of all behaviors
- Automatic detection of scene changes

---

## 🎛️ API Reference

### `extract_frames_kmeans_gpu()`

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
    max_iter: int = 100,
    random_state: int = 42,
    chunk_size: int = None
) -> Tuple[List[str], Dict[str, float]]
```

**Key Parameters:**
- `video_path`: Path to input video
- `output_dir`: Directory for extracted frames
- `n_frames`: Number of representative frames to extract
- `start_time`/`end_time`: Time range as fraction (0.0-1.0)
- `step`: Sample every Nth frame (higher = faster)
- `resize_width`: Thumbnail size for clustering (smaller = faster)
- `use_color`: RGB (True) or grayscale (False) clustering

**Returns:**
- `frames`: List of extracted frame paths
- `timing`: Dict with `'step1_time'`, `'step2_time'`, `'step3_time'`, `'total_time'`

### `benchmark_cpu_vs_gpu()`

```python
benchmark_cpu_vs_gpu(
    video_path: str,
    output_dir_gpu: str = "./gpu_output",
    output_dir_cpu: str = "./cpu_output",
    n_frames: int = 50,
    step: int = 5,
    **kwargs
) -> Dict
```

**Returns:**
Dictionary with detailed timing and speedup metrics:
- `'speedup'`: Overall speedup factor
- `'speedup_step1'`: Frame extraction speedup
- `'speedup_step2'`: K-means clustering speedup
- `'speedup_step3'`: Full-res export speedup
- `'cpu_timing'`: CPU timing breakdown
- `'gpu_timing'`: GPU timing breakdown

### `get_video_info()`

```python
get_video_info(video_path: str) -> Dict
```

Returns: `{'nframes', 'fps', 'width', 'height', 'duration'}`

---

## 🐛 Troubleshooting

### CUDA Not Available
```bash
nvidia-smi  # Check driver
nvcc --version  # Check CUDA
pip install cupy-cuda12x cuml-cu12 nvidia-dali-cuda120  # Reinstall
```

### Out of Memory
```python
# Reduce chunk size or thumbnail size
pyCAFE.extract_frames_kmeans_gpu(
    "video.mp4", "./frames",
    chunk_size=1000,  # Smaller chunks
    resize_width=20,  # Smaller thumbnails
    step=20           # Sample less
)
```

### Fewer Frames Than Requested

If you request more clusters than available unique frames, K-means may create empty clusters. This is expected for short videos or aggressive sampling.

**Solutions:**
- Reduce `n_frames` to match video length
- Decrease `step` to sample more frames
- Check video has sufficient visual diversity

### Video Format Issues
```bash
# Convert to H.264
ffmpeg -i input.avi -c:v libx264 -preset fast output.mp4
```

---

## 📚 Citation

```bibtex
@software{pycafe2025,
  title = {pyCAFE: Python CUDA Accelerated Frame Extractor},
  author = {Tan, Wulin},
  year = {2025},
  url = {https://github.com/Wulin-Tan/pyCAFE}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

- **[DeepLabCut](http://www.mackenziemathislab.org/deeplabcut)** - Markerless pose estimation
- **[SLEAP](https://sleap.ai/)** - Multi-animal pose tracking
- **[NVIDIA DALI](https://github.com/NVIDIA/DALI)** - GPU video decoding
- **[RAPIDS cuML](https://github.com/rapidsai/cuml)** - GPU machine learning

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/Wulin-Tan/pyCAFE/issues)
- **Email**: wulintan9527@gmail.com

---

<div align="center">

**⭐ Star this repo if pyCAFE helps your research! ⭐**

[🐛 Report Bug](https://github.com/Wulin-Tan/pyCAFE/issues) · [✨ Request Feature](https://github.com/Wulin-Tan/pyCAFE/issues)

</div>
