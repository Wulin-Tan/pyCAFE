"""
pyCAFE: Python CUDA Accelerated Frame Extractor
GPU-accelerated video frame extraction with K-means clustering
"""

__version__ = "0.1.0"
__author__ = "Wulin Tan"
__email__ = "wulintan9527@gmail.com"

# Check CUDA availability
try:
    from cuml.cluster import KMeans as cuKMeans
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from .core import (
    extract_frames_kmeans_gpu,
    extract_frames_kmeans_cpu,
)
from .extraction import (
    extract_downsampled_frames_gpu,
    extract_specific_frames,
    extract_and_save_fullres_frames,
)
from .utils import get_video_info
from .benchmark import benchmark_cpu_vs_gpu
from .clustering import perform_kmeans_gpu

__all__ = [
    'extract_frames_kmeans_gpu',
    'extract_frames_kmeans_cpu',
    'extract_downsampled_frames_gpu',
    'extract_specific_frames',
    'extract_and_save_fullres_frames',
    'get_video_info',
    'benchmark_cpu_vs_gpu',
    'perform_kmeans_gpu',
    'CUDA_AVAILABLE',
    '__version__',
]
