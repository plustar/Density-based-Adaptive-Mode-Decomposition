# tfvmd/core/config.py

from dataclasses import dataclass
from typing import Optional, Literal, Union, List
import numpy as np
import scipy.signal

@dataclass
class VMDConfig:
    """Configuration parameters for VMD algorithm."""
    num_channels: int
    n_fft: int = 128
    alpha: float = 10
    init: int = 1
    tol: float = 1e-5
    tau: float = 0.1
    win_exp: float = 1
    max_iterations: int = 1000
    modulated: bool = True
    sync: bool = False
    window_func: Optional[np.ndarray] = None
    rho_thresh: float = 0.1
    method: Literal['auto', 'imr', 'dme'] = 'dme'
    keep_residual: bool = False  # New parameter for DCME residual control
    
    def __post_init__(self):
        """Setup derived parameters after initialization."""
        self.padwidth = ((self.n_fft-1)//2, (self.n_fft-1)//2) \
            if (self.n_fft-1)%2==0 else ((self.n_fft-1)//2+1, (self.n_fft-1)//2)
        
        if self.window_func is None:
            self.window = scipy.signal.windows.hann(self.n_fft, sym=True)
        else:
            self.window = self.window_func

@dataclass
class BandwidthConfig:
    """Configuration for bandwidth estimation."""
    method: Literal['silverman', 'scott', 'percentile', 'adaptive'] = 'adaptive'
    scale_factor: float = 1.0
    base_method: Literal['silverman', 'scott'] = 'silverman'
    min_bandwidth: float = 1e-6

@dataclass
class ProcessingResult:
    """Container for processing results with time-frequency information."""
    tf_map: np.ndarray
    mode_functions: List[np.ndarray]
    time_indices: List[int]
    center_frequencies: List[float]
    residual: np.ndarray
    convergence_history: List[float]
    time_domain_modes: Optional[List[np.ndarray]] = None
    method_used: str = ''
    bandwidth_values: Optional[np.ndarray] = None

    def __post_init__(self):
        """Initialize additional attributes."""
        self.n_modes = len(self.mode_functions)
        self.time_range = len(np.unique(self.time_indices))
        
    def compute_tf_energy(self, freq_bins: int = None) -> np.ndarray:
        """Compute time-frequency energy distribution."""
        if freq_bins is None:
            freq_bins = self.mode_functions[0].shape[1]
            
        time_points = self.time_range
        tf_energy = np.zeros((freq_bins, time_points))
        
        unique_times = np.unique(self.time_indices)
        time_to_idx = {t: idx for idx, t in enumerate(unique_times)}
        
        max_freq = max(self.center_frequencies)
        freq_bin_width = max_freq / (freq_bins - 1)
        
        for mode, t, freq in zip(self.mode_functions, self.time_indices, 
                               self.center_frequencies):
            if isinstance(mode, np.ndarray):
                if mode.ndim == 2:
                    energy = np.mean(np.abs(mode)**2)
                elif mode.ndim == 3:
                    energy = np.mean(np.abs(mode)**2)
                else:
                    continue
            else:
                continue
                
            freq_idx = int(np.clip(freq / freq_bin_width, 0, freq_bins - 1))
            t_idx = time_to_idx[t]
            tf_energy[freq_idx, t_idx] += energy
            
        return tf_energy