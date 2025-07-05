# DAMD - Density-based Adaptive Mode Decomposition

![Python](https://img.shields.io/badge/python-3.8--3.9-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive Python implementation of Density-based Adaptive Mode Decomposition (DAMD), which addresses fundamental limitations in traditional Variational Mode Decomposition (VMD) through theoretical equivalence with density-based clustering and automatic parameter determination.

## 🌟 Features

- **Multiple Decomposition Methods**:
  - **DME (Direct Mode Extraction)**: Fast frequency-domain decomposition using clustering
  - **IMR (Interaction Mode Refinement)**: Variational decomposition for overlapping modes  
  - **HVR (Hybrid Variational Refinement)**: Combines DME initialization with selective VMD refinement
  - **VMD (Traditional)**: Classic Variational Mode Decomposition

- **Automatic Mode Detection**: Eliminates manual specification of mode numbers through meanshift clustering

- **Theoretical Foundation**: Mathematical equivalence between VMD optimization and density-based clustering

- **Adaptive Bandwidth Estimation**:
  - Silverman's rule of thumb
  - Scott's rule
  - Percentile-based estimation
  - Adaptive spectral curvature-based bandwidth

- **Flexible Time-Frequency Transforms**:
  - Short-Time Fourier Transform (STFT)
  - Synchrosqueezed STFT
  - Continuous Wavelet Transform (CWT)
  - Synchrosqueezed CWT

- **Advanced Visualization**:
  - Three-panel analysis plots
  - Mode center frequency tracking
  - Signal reconstruction comparison
  - Customizable styling and export options

## 🚀 Installation

### Prerequisites

- Python 3.8-3.9
- NumPy
- SciPy
- Matplotlib
- Numba
- ssqueezepy
- tqdm

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/plustar/Density-based-Adaptive-Mode-Decomposition.git
cd Density-based-Adaptive-Mode-Decomposition

# Create conda environment from provided file
conda env create -f environment.yml
conda activate acvmd
```

### Using pip

```bash
pip install numpy scipy matplotlib numba ssqueezepy tqdm
git clone https://github.com/plustar/Density-based-Adaptive-Mode-Decomposition.git
cd Density-based-Adaptive-Mode-Decomposition
```

## 📖 Quick Start

### Basic Usage

```python
import numpy as np
from tfvmd.core.decomposition import TimeFrequencyVMD
from tfvmd.core.config import VMDConfig, BandwidthConfig
from tfvmd.visualization.plotter import VMDVisualizer

# Generate a test signal
fs = 512
t = np.linspace(0, 2, fs*2)
signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.1*np.random.randn(len(t))

# Configure bandwidth estimation
bandwidth_config = BandwidthConfig(
    method='adaptive',
    scale_factor=1.0,
    base_method='silverman'
)

# Configure DAMD
vmd_config = VMDConfig(
    num_channels=1,
    n_fft=fs//2,
    alpha=50,
    method='dme',  # Choose specific method: 'dme', 'imr', 'hvr', 'vmd'
    max_iterations=1000
)

# Initialize and run decomposition
vmd = TimeFrequencyVMD(vmd_config, bandwidth_config)
signal_expanded = signal.reshape(1, -1)
tf_map, _ = vmd.transformer.transform(signal_expanded, transform_type='stft')
result = vmd.decompose(tf_map)

# Generate time-domain modes
result = vmd.generate_time_domain_modes(result, transform_type='stft')

# Visualize results
visualizer = VMDVisualizer()
fig = visualizer.plot_analysis(result, signal, fs)
plt.show()
```

### Advanced Configuration

```python
from tfvmd.visualization.config import VisualizationConfig, FontConfig, SaveConfig

# Custom visualization settings
viz_config = VisualizationConfig(
    base_fonts=FontConfig(
        base_font=12,
        base_title=14,
        base_label=12
    ),
    font_scale=1.2,
    color_palette='viridis',
    save_config=SaveConfig(
        format='svg',
        dpi=300,
        transparent=True
    )
)

visualizer = VMDVisualizer(viz_config)

# Create and save publication-ready figures
fig = visualizer.plot_and_save(
    result, signal, fs, 
    filename='decomposition_result',
    figsize=(12, 8)
)
```

## 🔧 Configuration Options

### VMD Configuration

```python
VMDConfig(
    num_channels=1,          # Number of signal channels
    n_fft=128,              # FFT size for transforms
    alpha=10,               # Balancing parameter for VMD
    tol=1e-5,              # Convergence tolerance
    tau=0.1,               # Lagrangian parameter
    max_iterations=1000,    # Maximum iterations
    method='dme',           # 'dme', 'imr', 'hvr', 'vmd'
    keep_residual=False     # Whether to keep residual in DME
)
```

### Bandwidth Configuration

```python
BandwidthConfig(
    method='adaptive',       # 'silverman', 'scott', 'percentile', 'adaptive'
    scale_factor=1.0,       # Scaling factor for bandwidth
    base_method='silverman', # Base method for adaptive estimation
    min_bandwidth=1e-6      # Minimum allowed bandwidth
)
```

## 📊 Method Selection Guide

| Method | Best For | Computational Cost | Accuracy |
|--------|----------|-------------------|----------|
| **DME** | Well-separated modes, fast processing | Low | Good |
| **IMR** | Overlapping modes, high precision | High | Excellent |
| **HVR** | Mixed scenarios, balanced performance | Medium | Very Good |

## 🎯 Key Features

- **Automatic Mode Detection**: Eliminates the need for manual specification of mode numbers through meanshift clustering
- **Computational Efficiency**: DME provides significant speed improvements over traditional VMD  
- **Theoretical Foundation**: Mathematical equivalence between VMD optimization and density-based clustering
- **Flexible Implementation**: Multiple approaches (DME, HVR) to balance efficiency and precision
- **Diverse Time-Frequency Support**: Works with STFT, CWT, synchrosqueezed transforms, and more
- **Real-world Validation**: Demonstrated effectiveness across biomedical, mechanical, audio, and gravitational wave signals

## 🎯 Applications

- **Biomedical Signal Processing**: EEG, ECG, EMG analysis with automatic mode detection
- **Mechanical Vibration Analysis**: Fault detection and modal analysis without prior parameter knowledge  
- **Audio Processing**: Speech enhancement and music analysis with adaptive decomposition
- **Gravitational Wave Detection**: Chirp pattern tracking and merger signature identification
- **Financial Time Series**: Trend and cycle extraction with time-varying complexity

## 📈 Key Results

DAMD provides comprehensive analysis through:

1. **Automatic Mode Detection**: Determines optimal number of modes without manual specification
2. **Mode Center Frequency Tracking**: Tracks temporal evolution of spectral components 
3. **Perfect Reconstruction**: Maintains signal fidelity through theoretical guarantees
4. **Computational Efficiency**: Significant speed improvements over traditional VMD (up to 5x faster with DME)
5. **Adaptive Bandwidth**: Automatically adjusts to local spectral density characteristics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the Variational Mode Decomposition algorithm by Dragomiretskiy & Zosso (2014)
- Implements theoretical framework from "Density-based Adaptive Mode Decomposition" research
- Uses [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy) for time-frequency transforms
- Visualization inspired by scientific plotting best practices

## 📫 Contact

- **Author**: Hao Jia
- **Email**: haojia@nankai.edu.cn  
- **GitHub**: [@plustar](https://github.com/plustar)
- **Institution**: School of Medicine, Nankai University

---

**Keywords**: signal processing, time-frequency analysis, variational mode decomposition, adaptive clustering, density-based clustering, automatic mode detection, non-stationary signals
