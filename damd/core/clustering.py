# damd/core/clustering.py

import numpy as np
from numba import jit, float64
from typing import Optional
from ..core.config import BandwidthConfig
from ..utils.bandwidth import estimate_bandwidth

@jit(float64[:](float64[:], float64[:], float64[:], float64[:]), nopython=True)
def shift_points(points: np.ndarray, frequencies: np.ndarray, 
                 powers: np.ndarray, bandwidths: np.ndarray) -> np.ndarray:
    """
    Shift multiple points towards mode centers with position-dependent bandwidth.
    
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
                seed_spacing: int = 5) -> np.ndarray:
        """
        Perform meanshift clustering with adaptive or fixed bandwidth.
        
        Args:
            freqs: Frequency points
            powers: Power/amplitude at each frequency
            t: Time index (for adaptive bandwidth)
            tf_map: Time-frequency representation (for adaptive bandwidth)
            max_iterations: Maximum number of iterations
            convergence_thresh: Convergence threshold
            seed_spacing: Spacing between seed points for uniform sampling (default: 3)
            
        Returns:
            Array of cluster centers
        """
        seed_spacing = self.bandwidth_config.seed_spacing
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
            
        # 使用均匀采样选择种子点
        seed_indices = np.arange(0, n_points, seed_spacing)
        positions = freqs[seed_indices].copy()
        
        # 为种子点准备对应的带宽
        seed_bandwidth = bandwidth[seed_indices].copy()
        
        # Meanshift迭代
        for _ in range(max_iterations):
            new_positions = shift_points(
                positions, freqs, powers, seed_bandwidth
            )
            
            # 检查收敛性
            if np.max(np.abs(new_positions - positions)) < convergence_thresh:
                break
                
            positions = new_positions
        
        # 提取聚类中心
        mean_bandwidth = np.mean(bandwidth)
        sorted_positions = np.sort(positions)
        position_diff = np.diff(sorted_positions)
        cluster_breaks = np.where(position_diff > mean_bandwidth)[0] + 1
        
        if len(cluster_breaks) == 0:
            cluster_centers = np.array([np.mean(positions)])
        else:
            splits = np.split(sorted_positions, cluster_breaks)
            cluster_centers = np.array([np.mean(cluster) for cluster in splits])
        
        # 计算每个聚类中心的功率并排序
        cluster_powers = np.array([
            np.max(powers[np.argmin(np.abs(freqs[:, np.newaxis] - center), axis=0)])
            for center in cluster_centers
        ])
        
        return cluster_centers[np.argsort(cluster_powers)[::-1]]