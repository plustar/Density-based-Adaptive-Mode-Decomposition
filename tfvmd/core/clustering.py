# tfvmd/core/clustering.py

import numpy as np
from numba import jit, float64
from typing import Optional
from ..core.config import BandwidthConfig
from ..utils.bandwidth import estimate_bandwidth

@jit(float64[:](float64[:], float64), nopython=True)
def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute Gaussian kernel."""
    return np.exp(-0.5 * (distances / bandwidth) ** 2)

@jit(float64(float64, float64[:], float64[:], float64), nopython=True)
def shift_point(point: float, frequencies: np.ndarray, 
                powers: np.ndarray, bandwidth: float) -> float:
    """Shift a single point towards mode center."""
    distances = np.abs(frequencies - point)
    kernel_weights = gaussian_kernel(distances, bandwidth)
    total_weights = kernel_weights * powers
    
    weight_sum = np.sum(total_weights)
    if weight_sum == 0:
        return point
        
    return np.sum(frequencies * total_weights) / weight_sum

@jit(float64[:](float64[:], float64[:], float64[:], float64[:]), nopython=True)
def batch_shift_points(points: np.ndarray, frequencies: np.ndarray, 
                      powers: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
    """
    Shift multiple points in parallel with position-dependent bandwidth.
    
    Args:
        points: Points to shift
        frequencies: Frequency points
        powers: Power at each frequency
        bandwidths: Bandwidth for each point (array)
        
    Returns:
        New positions after shifting
    """
    new_positions = np.empty_like(points)
    for i in range(len(points)):
        h = bandwidths[i]
        point = points[i]
        distances = np.abs(frequencies - point)
        kernel_weights = np.exp(-0.5 * (distances / h) ** 2)
        total_weights = kernel_weights * powers
        
        weight_sum = np.sum(total_weights)
        if weight_sum == 0:
            new_positions[i] = point
        else:
            new_positions[i] = np.sum(frequencies * total_weights) / weight_sum
            
    return new_positions

class MeanshiftClustering:
    """Implements meanshift clustering for frequency identification."""
    
    def __init__(self, bandwidth_config: BandwidthConfig):
        self.bandwidth_config = bandwidth_config

    def cluster(self, freqs: np.ndarray, powers: np.ndarray, 
                t: Optional[int] = None,
                tf_map: Optional[np.ndarray] = None,
                max_iterations: int = 300,
                convergence_thresh: float = 1e-4,
                batch_size: int = 50) -> np.ndarray:
        """
        Perform meanshift clustering with adaptive or fixed bandwidth.
        
        Args:
            freqs: Frequency points
            powers: Power/amplitude at each frequency
            t: Time index (for adaptive bandwidth)
            tf_map: Time-frequency representation (for adaptive bandwidth)
            max_iterations: Maximum number of iterations
            convergence_thresh: Convergence threshold
            batch_size: Size of batches for processing
            
        Returns:
            Array of cluster centers
        """
        freqs = np.ascontiguousarray(freqs, dtype=np.float64)
        powers = np.ascontiguousarray(powers / np.max(powers), dtype=np.float64)
        n_points = len(freqs)
        
        bandwidth = estimate_bandwidth(
            freqs, powers,
            config=self.bandwidth_config,
            t=t, tf_map=tf_map
        )
        
        if np.isscalar(bandwidth):
            bandwidth = np.full(n_points, bandwidth, dtype=np.float64)
        else:
            bandwidth = np.ascontiguousarray(bandwidth, dtype=np.float64)
            
        final_positions = np.copy(freqs)
        
        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            batch_positions = final_positions[start_idx:end_idx].copy()
            batch_bandwidth = bandwidth[start_idx:end_idx].copy()
            
            for _ in range(max_iterations):
                new_positions = batch_shift_points(
                    batch_positions, freqs, powers, batch_bandwidth
                )
                
                if np.max(np.abs(new_positions - batch_positions)) < convergence_thresh:
                    break
                    
                batch_positions = new_positions
            
            final_positions[start_idx:end_idx] = batch_positions
        
        mean_bandwidth = np.mean(bandwidth)
        sorted_positions = np.sort(final_positions)
        position_diff = np.diff(sorted_positions)
        cluster_breaks = np.where(position_diff > mean_bandwidth)[0] + 1
        
        if len(cluster_breaks) == 0:
            cluster_centers = np.array([np.mean(final_positions)])
        else:
            splits = np.split(sorted_positions, cluster_breaks)
            cluster_centers = np.array([np.mean(cluster) for cluster in splits])
        
        cluster_powers = np.array([
            np.max(powers[np.argmin(np.abs(freqs[:, np.newaxis] - center), axis=0)])
            for center in cluster_centers
        ])
        
        return cluster_centers[np.argsort(cluster_powers)[::-1]]