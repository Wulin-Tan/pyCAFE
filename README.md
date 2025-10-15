# pyCAFE: Python CUDA Accelerated Frame Extractor

GPU-accelerated video frame extraction using K-means clustering for intelligent frame selection.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-11.x%2F12.x-brightgreen)

## 🚀 Features

- **GPU-Accelerated Processing**: Leverages NVIDIA DALI for fast video decoding and preprocessing
- **Intelligent Frame Selection**: Uses K-means clustering (cuML) to select representative frames
- **Automatic Chunking**: Handles long videos efficiently with automatic chunk-based processing
- **Flexible Configuration**: Customizable parameters for frame sampling, clustering, and output
- **CPU Fallback**: Automatically falls back to CPU mode if GPU libraries aren't available
- **Benchmarking Tools**: Built-in CPU vs GPU performance comparison
- **Command-Line Interface**: Easy-to-use CLI for quick operations
- **Python API**: Comprehensive API for integration into pipelines

## 📋 Requirements

### Minimum Requirements
- Python 3.8+
- OpenCV
- NumPy
- scikit-learn
- Pillow
- tqdm

### GPU Requirements (Optional but Recommended)
- NVIDIA GPU with CUDA support
- CUDA 11.x or 12.x
- cuPy (CUDA arrays)
- cuML (GPU K-means)
- NVIDIA DALI (GPU video decoding)

## 🔧 Installation

### Basic Installation (CPU only)

```bash  
pip install pyCAFE
