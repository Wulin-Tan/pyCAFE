---
title: 'pyCAFE: CUDA-Accelerated Frame Extraction for Behavioral Video Analysis'
tags:
  - Python
  - CUDA
  - behavioral neuroscience
  - computer vision
  - pose estimation
  - video processing
  - GPU acceleration
authors:
  - name: Wulin Tan
    orcid: 0000-0003-1678-1094
    corresponding: true
    affiliation: 1
  - name: Xiang Gao
    orcid: 0000-0001-9277-1432
    affiliation: 1
affiliations:
 - name: First Affiliated Hospital of Sun Yat-sen University, Guangzhou, China
   index: 1
date: 26 October 2024
bibliography: paper.bib
---

# Summary

Modern behavioral neuroscience relies on markerless pose estimation tools like DeepLabCut and SLEAP that require diverse training frames extracted from hours of video data. Existing frame extraction tools use K-means clustering for intelligent selection but are CPU-based and slow for large datasets. We present pyCAFE, a GPU-accelerated frame extractor that achieves 4-40× speedup over CPU approaches by leveraging NVIDIA DALI for video decoding and RAPIDS cuML for clustering. The software is open-source, memory-efficient, and integrates seamlessly with existing pose estimation workflows.

# Statement of Need

Behavioral neuroscience increasingly depends on deep learning-based pose estimation tools [@Mathis2018; @Pereira2022] that track animal movement without physical markers. Training these models requires extracting diverse, representative frames from behavioral videos—often hours long and totaling terabytes of data [@Datta2019].

This preprocessing bottleneck directly impacts research throughput: laboratories conducting multi-animal studies, longitudinal experiments, or high-throughput phenotyping screens routinely process hundreds of videos. The time cost of frame extraction compounds across experimental pipelines, delaying downstream analysis and limiting the scale of feasible studies. For instance, extracting training frames from 100 videos in a typical study can require 4-6 hours of CPU processing time, creating a significant barrier to rapid iteration and large-scale behavioral analysis.

DeepLabCut employs K-means clustering for frame selection to ensure visual diversity [@Mathis2018], but this approach becomes computationally expensive for large datasets. For a typical 30-minute video with 54,000 frames, CPU-based K-means clustering can take 10-15 seconds just for the clustering step, with total extraction time exceeding 2 minutes per video. When processing dozens or hundreds of videos, this becomes a major bottleneck.

Existing solutions lack GPU acceleration: DeepLabCut's extractor is CPU-based, SLEAP uses uniform sampling without intelligent clustering, and generic tools like FFmpeg cannot perform K-means-based frame selection. pyCAFE addresses this gap by accelerating both video decoding and clustering on GPU, reducing processing time by 4-40× while maintaining the quality benefits of K-means-based selection.

# Implementation

pyCAFE implements a three-stage GPU-accelerated pipeline:

**Frame Extraction:** NVIDIA DALI [@DALI2021] performs hardware-accelerated video decoding and resizes frames to thumbnails (default 30×N pixels). This reduces memory footprint 100-1000× while preserving visual information sufficient for behavioral pose discrimination.

**K-means Clustering:** RAPIDS cuML [@RAPIDS2023] performs GPU-accelerated K-means clustering on thumbnail feature vectors. The algorithm groups visually similar frames and selects one representative per cluster, ensuring both temporal and visual diversity.

**Full-Resolution Export:** OpenCV extracts selected frame indices at original resolution from the source video, ensuring output quality is unaffected by thumbnail-based clustering.

The software automatically chunks large videos to fit GPU memory and provides CPU fallback for systems without GPU access. All operations are reproducible through seeded random states.

**Software Architecture:**

- Core dependencies: NumPy, OpenCV, scikit-learn
- GPU acceleration: CuPy, RAPIDS cuML 23.x, NVIDIA DALI
- Platform support: Linux (full GPU), Windows (via WSL2), macOS (CPU only)
- License: MIT
- Installation: PyPI and Conda

# Usage Example

```python
import pyCAFE

# Extract 100 diverse frames using GPU acceleration
frames, timing = pyCAFE.extract_frames_kmeans_gpu(
    video_path="behavior.mp4",
    output_dir="./frames",
    n_frames=100,
    step=5
)
```

The software integrates directly with DeepLabCut and SLEAP workflows by exporting frames in their expected format.

# Performance Benchmarks

We evaluated pyCAFE on three publicly available behavioral neuroscience datasets [@Forkosh2019; @Klibaite2021; @Jhuang2010] using an NVIDIA Tesla T4 GPU (16GB) with Intel Xeon 8-core CPU:

|Video Duration|Resolution|Total Frames|CPU Time|GPU Time|Speedup|
|---|---|---|---|---|---|
|1 min|1280×720|1,981|29.5s|2.0s|14.7×|
|15 min|640×480|27,000|66.7s|9.5s|7.0×|
|55 min|480×480|99,150|144.5s|36.7s|3.9×|

K-means clustering showed the most dramatic acceleration (57-134× depending on video length), with speedup increasing for longer videos where CPU memory bandwidth becomes severely bottlenecked. For the 55-minute video, clustering alone took 24.9s on CPU versus 0.2s on GPU—a 134× improvement.

GPU advantage increases dramatically with larger thumbnail sizes: using 256px width thumbnails on the 55-minute video achieved 39.7× overall speedup (4713s CPU vs 123s GPU for clustering), demonstrating that GPU parallelism scales more efficiently than CPU with increasing feature dimensionality. However, the default 30px setting provides sufficient discrimination for behavioral poses while maximizing speed.

# Practical Impact

For a typical behavioral study processing 20 videos of 30 minutes each:

- **Traditional approach:** ~2-3 min/video × 20 = 40-60 minutes
- **pyCAFE (default settings):** ~20-30 sec/video × 20 = **7-10 minutes**

This 6-8× reduction in processing time enables rapid iteration on frame selection parameters, processing of entire experimental batches in minutes, and feasibility of large-scale behavioral phenotyping studies. The efficiency gains make sophisticated frame selection accessible to laboratories without dedicated computing clusters.

Frame quality validation by three neuroscience researchers confirmed that pyCAFE's K-means selection captured all major behavioral states (grooming, rearing, locomotion, resting), avoided redundant frames during static periods, and produced training sets rated as more diverse compared to uniform or random sampling.

# Comparison with Existing Tools

|Tool|GPU Accelerated|Intelligent Selection|Batch Processing|
|---|---|---|---|
|DeepLabCut Extractor|No|Yes (K-means)|No|
|SLEAP Extractor|No|No (uniform)|Limited|
|FFmpeg|Partial*|No|Yes|
|**pyCAFE**|**Yes**|**Yes**|**Yes**|

*FFmpeg can use GPU decoders but not for clustering.

# Use Cases

pyCAFE is designed for behavioral neuroscience workflows including:

- Extracting training frames for DeepLabCut/SLEAP pose estimation
- Behavioral classification and ethogram development  
- Quality control sampling for annotation validation
- Exploratory analysis of long recordings

The software has been successfully used in pilot studies involving mouse social interactions, zebrafish optomotor responses, and rat navigation tasks.

# Future Directions

Planned enhancements include:

- Automatic cluster number selection (elbow method, silhouette analysis)
- ROI-based clustering (focus on animal position, ignore background)
- Alternative clustering methods (DBSCAN, hierarchical)
- Direct DeepLabCut/SLEAP configuration file integration

# Availability

- **Source code:** https://github.com/Wulin-Tan/pyCAFE
- **Documentation:** https://github.com/Wulin-Tan/pyCAFE#readme
- **PyPI:** Package available for installation via pip
- **Tests:** Included in repository with pytest suite

# Acknowledgments

We thank the authors of publicly available behavioral video datasets [@Forkosh2019; @Klibaite2021; @Jhuang2010] used in benchmarking.

# References
