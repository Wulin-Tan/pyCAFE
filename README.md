# pyCAFE: Python CUDA Accelerated Frame Extractor

**GPU-accelerated video frame extraction with intelligent K-means clustering for neuroscience and behavioral analysis**

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-11.x%2F12.x-brightgreen)

---

## 🧠 Overview

pyCAFE extracts diverse, representative frames from behavioral videos using GPU-accelerated K-means clustering. Perfect for creating training datasets for DeepLabCut, SLEAP, and other pose estimation tools.

**Why pyCAFE?** K-means clustering ensures visual and temporal diversity (similar to DeepLabCut's frame selection), but **7-40x faster** with GPU acceleration.

### Key Features

- ⚡ **7-40x faster** than CPU processing
- 🔬 **Smart K-means clustering** for diverse frame selection
- 📊 **Memory efficient** - handles videos of any length
- 💻 **CPU fallback** when GPU unavailable
- 🎯 **Full resolution output** - thumbnails only used for clustering speed

---

## 🚀 Quick Start

### Installation

```bash
# CPU only (slower)
pip install pyCAFE

# GPU acceleration (recommended) - CUDA 12.x
pip install pyCAFE cupy-cuda12x cuml-cu12
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120
```

<details>
<summary>📦 Alternative installation methods (CUDA 11.x / Conda)</summary>

**CUDA 11.x:**
```bash
pip install pyCAFE cupy-cuda11x cuml-cu11
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
```

**Conda (easiest):**
```bash
conda create -n pycafe python=3.10 -y
conda activate pycafe
conda install -c rapidsai -c conda-forge -c nvidia cuml=23.10 cupy cudatoolkit=11.8
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
pip install pyCAFE
```
</details>

### Basic Usage

**Command Line:**
```bash
# Extract 50 diverse frames
pycafe video.mp4 -o ./frames -n 50

# Extract from middle 80% of video (skip intro/outro)
pycafe video.mp4 -o ./frames -n 100 --start 0.1 --end 0.9 --step 5

# Benchmark your video
pycafe video.mp4 --benchmark -n 50 --step 5
```

**Python API:**
```python
import pyCAFE

# Extract frames
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="behavior.mp4",
    output_dir="./frames",
    n_frames=100,  # Number of frames to extract
    step=5         # Sample every 5th frame (faster)
)

print(f"✅ Extracted {len(frames)} frames in {timing['total_time']:.1f}s")
```

---

## 📊 Performance Benchmarks

### Overall Performance (Default Settings)

**Configuration:** 50 frames, `step=5`, `resize_width=30` (default)

| Video | Duration | Resolution | Total Frames | CPU Time | GPU Time | **Speedup** | Dataset Source |
|-------|----------|------------|--------------|----------|----------|-------------|----------------|
| Mouse | 1 min | 1280×720 | 1,981 | 29.5s | 2.0s | **14.7x** | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.mw6m905v3) |
| Open Field | 15 min | 640×480 | 27,000 | 66.7s | 9.5s | **7.0x** | [Zenodo](https://zenodo.org/records/4629544) |
| Long Recording | 55 min | 480×480 | 99,150 | 144.5s | 36.7s | **3.9x** | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FSAPNJG) |

<details>
<summary>📈 Step-by-Step Performance Breakdown</summary>

**1-minute video (1280×720, 1,981 frames):**

| Processing Step | CPU Time | GPU Time | **Speedup** | % of Total Time (GPU) |
|-----------------|----------|----------|-------------|----------------------|
| Frame Extraction | 24.3s | 1.1s | **23.2x** | 53.4% |
| K-means Clustering | 0.4s | 0.1s | **4.1x** | 3.9% |
| Full-res Export | 4.8s | 0.9s | **5.6x** | 42.7% |
| **Total Pipeline** | **29.5s** | **2.0s** | **14.7x** | **100%** |

**15-minute video (640×480, 27,000 frames):**

| Processing Step | CPU Time | GPU Time | **Speedup** | % of Total Time (GPU) |
|-----------------|----------|----------|-------------|----------------------|
| Frame Extraction | 60.3s | 8.3s | **7.2x** | 87.4% |
| K-means Clustering | 5.4s | 0.1s | **56.9x** | 1.1% |
| Full-res Export | 1.1s | 1.1s | **1.0x** | 11.6% |
| **Total Pipeline** | **66.7s** | **9.5s** | **7.0x** | **100%** |

**55-minute video (480×480, 99,150 frames):**

| Processing Step | CPU Time | GPU Time | **Speedup** | % of Total Time (GPU) |
|-----------------|----------|----------|-------------|----------------------|
| Frame Extraction | 119.0s | 35.9s | **3.3x** | 97.8% |
| K-means Clustering | 24.9s | 0.2s | **134.3x** | 0.5% |
| Full-res Export | 0.7s | 0.7s | **1.0x** | 1.9% |
| **Total Pipeline** | **144.5s** | **36.7s** | **3.9x** | **100%** |

**Key Insights:**
- 🚀 K-means clustering: **4-134x faster** on GPU (nearly instantaneous)
- 📹 Frame extraction: **3-23x speedup** (higher resolution = greater benefit)
- 📊 High-res videos (1280×720) show dramatic 23x speedup vs. 3x for lower resolution
- ⏱️ Short videos: Export takes 43% of time due to seeking overhead
</details>

<details>
<summary>🔍 Impact of resize_width Parameter on Performance</summary>

The `resize_width` parameter controls thumbnail size for clustering. **Larger thumbnails dramatically increase GPU advantage:**

**1-minute video (1280×720, 1,981 frames):**

| `resize_width` | Thumbnail Size | CPU Total | GPU Total | **Speedup** | K-means (CPU) | K-means (GPU) |
|----------------|----------------|-----------|-----------|-------------|---------------|---------------|
| **30** (default) | 30×17px (~510 pixels) | 29.5s | 2.0s | **14.7x** | 0.4s | 0.1s |
| **256** (high-res) | 256×144px (~36,860 pixels) | 55.3s | 7.7s | **7.2x** | 25.4s | 4.1s |

**15-minute video (640×480, 27,000 frames):**

| `resize_width` | Thumbnail Size | CPU Total | GPU Total | **Speedup** | K-means (CPU) | K-means (GPU) |
|----------------|----------------|-----------|-----------|-------------|---------------|---------------|
| **30** (default) | 30×20px (~600 pixels) | 66.7s | 9.5s | **7.0x** | 5.4s | 0.1s |
| **256** (high-res) | 256×170px (~43,500 pixels) | 1054.9s | 31.5s | **33.5x** | 991.3s | 7.8s |

**55-minute video (480×480, 99,150 frames):**

| `resize_width` | Thumbnail Size | CPU Total | GPU Total | **Speedup** | K-means (CPU) | K-means (GPU) |
|----------------|----------------|-----------|-----------|-------------|---------------|---------------|
| **30** (default) | 30×30px (~900 pixels) | 144.5s | 36.7s | **3.9x** | 24.9s | 0.2s |
| **256** (high-res) | 256×256px (65,536 pixels) | 4881.9s | 122.9s | **39.7x** | 4713.0s | 41.5s |

**Summary:**
- **Short videos (1 min)**: Speedup decreases 14.7x → 7.2x with large thumbnails (initialization overhead)
- **Medium videos (15 min)**: Speedup increases 7.0x → 33.5x (CPU k-means bottleneck)
- **Long videos (55 min)**: Speedup increases 3.9x → 39.7x (maximum CPU bottleneck)

**Visualization:**
```
CPU vs GPU K-means Time

1-min (w=30):     CPU ▏ 0.4s      GPU ▏ 0.1s       (4x)
1-min (w=256):    CPU ██████▎ 25.4s    GPU █ 4.1s   (6x)

15-min (w=30):    CPU █▍ 5.4s     GPU ▏ 0.1s       (57x)
15-min (w=256):   CPU ████████████████████████████ 991.3s
                  GPU ██ 7.8s                       (127x)

55-min (w=30):    CPU ██████▎ 24.9s    GPU ▏ 0.2s  (134x)
55-min (w=256):   CPU ███████████████████████████████████████████ 4713.0s
                  GPU ██████████ 41.5s              (114x)
```

*Benchmarked on NVIDIA Tesla T4 (16GB) with Intel Xeon (Cascadelake) 8 vCPU @ 56GB RAM*
</details>

---

## ⚙️ Parameters & Configuration

### Common Parameters

```python
pyCAFE.extract_frames_kmeans_gpu(
    video_path="video.mp4",
    output_dir="./frames",
    n_frames=50,          # Number of output frames
    step=5,               # Sample every Nth frame (higher=faster)
    resize_width=30,      # Thumbnail size for clustering (default=30)
    start_time=0.0,       # Start fraction (0.0=beginning)
    end_time=1.0,         # End fraction (1.0=end)
    use_color=False,      # RGB or grayscale clustering
    random_state=42       # For reproducibility
)
```

<details>
<summary>📋 Parameter Guidelines by Video Length</summary>

| Parameter | Short Video (< 5 min) | Medium (10-30 min) | Long (> 1 hr) |
|-----------|----------------------|-------------------|---------------|
| `n_frames` | 30-50 | 50-100 | 100-300 |
| `step` | 2-5 | 5-10 | 20-30 |
| `resize_width` | 30 (default) | 30 (default) | 30 (default) |

**Quick recommendations:**
```python
import pyCAFE

info = pyCAFE.get_video_info("video.mp4")
duration = info['duration']

if duration < 300:  # < 5 min
    config = {'n_frames': 50, 'step': 5}
elif duration < 1800:  # 5-30 min
    config = {'n_frames': 100, 'step': 10}
else:  # > 30 min
    config = {'n_frames': 200, 'step': 20}

frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    "video.mp4", "./frames", **config
)
```
</details>

<details>
<summary>🔧 Understanding resize_width Parameter</summary>

**What it does:** Controls thumbnail size for clustering analysis only - **output frames are always full resolution**

**When to adjust:**

| Use Case | Recommended Value | Clustering Time | Best For |
|----------|-------------------|-----------------|----------|
| **Most behavioral videos** | **30** (default) ✅ | 0.1-0.2s GPU | Fastest, distinguishes poses effectively |
| **Small objects in frame** | 50-80 | 0.5-2s GPU | Fine detail discrimination |
| **Texture/pattern analysis** | 128-256 | 4-42s GPU | Research comparisons only |

**Important:** 
- For **short videos (< 5 min)**, large `resize_width` reduces GPU advantage (use default)
- Default (30px) is sufficient for most behavioral analysis
- Output frames are **always full resolution** regardless of this setting

**Example:**
```python
# Default: Fast and effective (recommended)
frames, t1 = pyCAFE.extract_frames_kmeans_gpu(
    "video.mp4", "./frames", n_frames=50
)
print(f"Default: {t1['total_time']:.1f}s")  # ~10s

# High-res: Better discrimination but much slower
frames, t2 = pyCAFE.extract_frames_kmeans_gpu(
    "video.mp4", "./frames_highres", n_frames=50,
    resize_width=128
)
print(f"High-res: {t2['total_time']:.1f}s")  # ~20s (CPU: 500s!)
```
</details>

---

## 💡 Example Use Cases

<details>
<summary>🔬 DeepLabCut Training Data Preparation</summary>

```python
import pyCAFE

# Extract 150 diverse frames for labeling
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="mouse_openfield.mp4",
    output_dir="./deeplabcut_frames",
    n_frames=150,
    step=10,
    start_time=0.1,  # Skip acclimation period
    end_time=0.95    # Skip removal period
)

print(f"✅ Ready for labeling: {len(frames)} frames in {timing['total_time']:.1f}s")
```
</details>

<details>
<summary>📦 Batch Processing Multiple Videos</summary>

```python
from pathlib import Path
import pandas as pd

video_dir = Path("./videos")
results = []

for video in video_dir.glob("*.mp4"):
    try:
        frames, timing = pyCAFE.extract_frames_kmeans_gpu(
            video_path=str(video),
            output_dir=f"./frames/{video.stem}",
            n_frames=100,
            step=10
        )
        results.append({
            'video': video.name,
            'frames': len(frames),
            'time': timing['total_time'],
            'status': 'success'
        })
        print(f"✅ {video.name}: {len(frames)} frames in {timing['total_time']:.1f}s")
    except Exception as e:
        results.append({
            'video': video.name,
            'frames': 0,
            'time': 0,
            'status': f'failed: {str(e)}'
        })
        print(f"❌ {video.name}: {str(e)}")

# Save report
df = pd.DataFrame(results)
df.to_csv("./frames/processing_report.csv", index=False)
print(f"\n📊 Processed {len(results)} videos")
```
</details>

<details>
<summary>⚡ Benchmark Your Own Videos</summary>

```python
import pyCAFE

# Compare CPU vs GPU performance
results = pyCAFE.benchmark_cpu_vs_gpu(
    video_path="your_video.mp4",
    n_frames=50,
    step=5
)

print(f"🎯 Overall Speedup: {results['speedup']:.2f}x")
print(f"   Frame Extraction: {results['speedup_step1']:.2f}x")
print(f"   K-means Clustering: {results['speedup_step2']:.2f}x")
print(f"   Full-res Export: {results['speedup_step3']:.2f}x")

# Test different resize_width values
for w in [30, 64, 128, 256]:
    frames, timing = pyCAFE.extract_frames_kmeans_gpu(
        "your_video.mp4", f"./test_w{w}",
        n_frames=50, resize_width=w
    )
    print(f"resize_width={w}: {timing['total_time']:.1f}s "
          f"(k-means: {timing['step2_time']:.1f}s)")
```
</details>

---

## 🔧 Troubleshooting

<details>
<summary>❌ GPU not detected</summary>

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall GPU packages (CUDA 12.x)
pip uninstall cupy cuml nvidia-dali -y
pip install cupy-cuda12x cuml-cu12
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120

# Verify
python -c "import cupy; import cuml; from nvidia.dali import pipeline; print('✅ GPU libraries loaded')"
```
</details>

<details>
<summary>💾 Out of memory error</summary>

```python
# Reduce memory usage
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    "video.mp4", "./frames",
    chunk_size=1000,    # Smaller chunks (default: auto)
    resize_width=20,    # Smaller thumbnails (default: 30)
    step=20             # More aggressive sampling (default: 1)
)
```
</details>

<details>
<summary>⚠️ Fewer frames than requested / Empty cluster warnings</summary>

**Cause:** Video is short, has low diversity, or `resize_width` too large

**Solutions:**
```python
# Option 1: Reduce n_frames
n_frames=20  # Instead of 50

# Option 2: Sample more frames
step=2  # Instead of 5

# Option 3: Reduce resize_width (especially for short videos)
resize_width=30  # Instead of 128 or 256
```

**Example:** The 1-min video with `resize_width=256` extracted only 12 frames (38 empty clusters) instead of 50, showing over-discrimination.
</details>

<details>
<summary>🎬 Video format issues</summary>

**Symptoms:** Cannot open video, decoding errors

**Solution:** Convert to H.264 MP4 (most compatible):
```bash
ffmpeg -i input.avi -c:v libx264 -preset fast -crf 23 output.mp4

# For large files (more compression)
ffmpeg -i input.avi -c:v libx264 -preset slow -crf 28 output.mp4

# Fix variable FPS
ffmpeg -i variable_fps.mp4 -vsync 1 -r 30 constant_fps.mp4
```
</details>

<details>
<summary>🐌 Slow performance on GPU</summary>

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi  # Should show 80-100% during extraction
```

**Solutions:**
1. **Use fast storage** (SSD/NVMe, not network drives)
2. **Check bottleneck:**
   ```python
   frames, timing = pyCAFE.extract_frames_kmeans_gpu("video.mp4", "./frames")
   
   if timing['step1_time'] > timing['step2_time'] * 10:
       print("⚠️ CPU bottleneck - video decoding is slow")
       print("Consider: lower resolution source, faster storage")
   ```
3. **Adjust chunk size:**
   ```python
   # Try different values: 1000-2000
   frames, timing = pyCAFE.extract_frames_kmeans_gpu(
       "video.mp4", "./frames", chunk_size=1500
   )
   ```
</details>

<details>
<summary>🎭 Poor frame diversity / Similar frames</summary>

**Solutions:**
```python
# Option 1: Increase thumbnail resolution
resize_width=50  # or 80 for subtle differences

# Option 2: Enable color clustering
use_color=True  # If color distinguishes behaviors

# Option 3: Sample more frames
step=5  # Instead of 10+

# Option 4: Adjust time range
start_time=0.2, end_time=0.95  # Skip boring sections
```
</details>

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

## 🔗 Links

- **Issues**: [GitHub Issues](https://github.com/Wulin-Tan/pyCAFE/issues)
- **Related**: [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) · [SLEAP](https://sleap.ai/) · [NVIDIA DALI](https://github.com/NVIDIA/DALI) · [RAPIDS cuML](https://github.com/rapidsai/cuml)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🧪 Testing
Run the test suite:

```bash
# Using unittest (built-in)
python tests/test_basic.py
```
---

<div align="center">

**⭐ Star this repo if pyCAFE helps your research! ⭐**

Made with ❤️ for the neuroscience community

</div>
