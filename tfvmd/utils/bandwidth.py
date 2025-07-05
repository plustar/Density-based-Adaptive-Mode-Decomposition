# tfvmd/utils/bandwidth.py

import numpy as np
from numba import jit, float64
from typing import Union, Optional
from ..core.config import BandwidthConfig

@jit(float64[:](float64[:], float64[:], float64[:]), nopython=True)
def compute_spectral_curvature(freqs: np.ndarray, powers: np.ndarray, 
                             weights: np.ndarray) -> np.ndarray:
    """
    Compute the second derivative of the weighted spectral density with stability checks.
    
    Args:
        freqs: Frequency points
        powers: Power/amplitude at each frequency
        weights: Weights for each frequency point
        
    Returns:
        Array of spectral curvature estimates
    """
    # Add small constant to prevent division by zero
    eps = 1e-10
    weighted_spectrum = powers * weights + eps
    
    df = max(freqs[1] - freqs[0], eps)  # Ensure non-zero frequency step
    df2 = df * df
    
    d2_spectrum = np.zeros_like(weighted_spectrum)
    
    # Interior points
    d2_spectrum[1:-1] = (weighted_spectrum[2:] - 2*weighted_spectrum[1:-1] + 
                        weighted_spectrum[:-2]) / df2
    
    # Boundary points with forward/backward differences
    d2_spectrum[0] = (weighted_spectrum[2] - 2*weighted_spectrum[1] + 
                     weighted_spectrum[0]) / df2
    d2_spectrum[-1] = (weighted_spectrum[-1] - 2*weighted_spectrum[-2] + 
                      weighted_spectrum[-3]) / df2
    
    # Ensure non-negative curvature and handle potential numerical issues
    return np.maximum(np.abs(d2_spectrum), eps)

def estimate_bandwidth(norm_freqs: np.ndarray, 
                      powers: np.ndarray,
                      config: BandwidthConfig,
                      t: Optional[int] = None,
                      tf_map: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
    """
    Estimate optimal bandwidth with robust numerical handling.
    
    Args:
        norm_freqs: Normalized frequencies
        powers: Power/amplitude at each frequency
        config: Bandwidth estimation configuration
        t: Time index (for adaptive method with tf_map)
        tf_map: Time-frequency representation (for adaptive method)
        
    Returns:
        Estimated bandwidth (scalar or array)
    """
    # Input validation and preprocessing
    if np.any(np.isnan(norm_freqs)) or np.any(np.isnan(powers)):
        raise ValueError("Input contains NaN values")
        
    norm_freqs = np.ascontiguousarray(norm_freqs, dtype=np.float64)
    powers = np.ascontiguousarray(powers, dtype=np.float64)
    
    # Normalize powers to prevent numerical issues
    powers = powers / (np.max(np.abs(powers)) + 1e-10)
    
    n_samples = len(norm_freqs)
    eps = 1e-10  # Small constant for numerical stability
    
    # Compute base bandwidth h0 using specified method
    if config.base_method == 'silverman':
        # Robust weighted mean and standard deviation calculation
        weighted_mean = np.average(norm_freqs, weights=powers+eps)
        weighted_var = np.average((norm_freqs - weighted_mean)**2, 
                                weights=powers+eps)
        weighted_std = np.sqrt(max(weighted_var, eps))
        h0 = 0.9 * weighted_std * n_samples**(-0.2)
        
    elif config.base_method == 'scott':
        weighted_mean = np.average(norm_freqs, weights=powers+eps)
        weighted_var = np.average((norm_freqs - weighted_mean)**2, 
                                weights=powers+eps)
        weighted_std = np.sqrt(max(weighted_var, eps))
        h0 = 3.49 * weighted_std * n_samples**(-1/3)
    
    else:
        raise ValueError(f"Unknown base bandwidth method: {config.base_method}")
    
    # Ensure minimum base bandwidth
    h0 = max(h0, config.min_bandwidth)
    
    # Apply selected bandwidth estimation method
    if config.method == 'adaptive':
        if tf_map is None or t is None:
            raise ValueError("tf_map and t required for adaptive bandwidth estimation")
        
        # Compute weights from tf_map with normalization
        if tf_map.ndim == 3:
            weights = np.abs(tf_map[:,:,t]).mean(0)
            weights = weights / (np.max(np.abs(weights)) + eps)
        else:
            weights = np.ones_like(powers)
            
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        
        # Compute curvature and adaptive bandwidth
        try:
            d2rho = compute_spectral_curvature(norm_freqs, powers, weights)
            
            # Ensure non-zero denominator and clip ratios
            min_d2rho = max(config.min_bandwidth, eps)
            scale_ratio = np.clip(config.scale_factor / (d2rho + min_d2rho),
                                eps, 1.0)
            
            bandwidth = h0 * np.sqrt(scale_ratio)
            
            # Ensure bandwidth is within reasonable bounds
            bandwidth = np.clip(bandwidth, 
                              config.min_bandwidth,
                              config.scale_factor * h0)
            
        except Exception as e:
            print(f"Warning: Adaptive bandwidth calculation failed at t={t}: {str(e)}")
            print("Falling back to base bandwidth")
            bandwidth = h0 * np.ones_like(norm_freqs)
        
    elif config.method == 'silverman':
        bandwidth = h0
        
    elif config.method == 'scott':
        bandwidth = h0
        
    elif config.method == 'percentile':
        sorted_diffs = np.sort(np.abs(np.diff(np.sort(norm_freqs))))
        valid_diffs = sorted_diffs[sorted_diffs > eps]
        if len(valid_diffs) > 0:
            bandwidth = np.percentile(valid_diffs, 10)
        else:
            bandwidth = h0
    
    else:
        raise ValueError(f"Unknown bandwidth estimation method: {config.method}")
    
    # Final validation and scaling
    bandwidth = np.asarray(bandwidth) * config.scale_factor
    
    # Ensure output is valid
    if np.any(np.isnan(bandwidth)) or np.any(np.isinf(bandwidth)):
        print(f"Warning: Invalid bandwidth values detected at t={t}")
        bandwidth = np.full_like(norm_freqs, h0 * config.scale_factor)
    
    return bandwidth