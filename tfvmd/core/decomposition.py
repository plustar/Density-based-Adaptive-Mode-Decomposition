# tfvmd/core/decomposition.py

from typing import Tuple, Optional, Union, List, Dict
import numpy as np
from tqdm import tqdm
import copy
import scipy
from scipy.interpolate import interp1d
from ..core.config import VMDConfig, ProcessingResult, BandwidthConfig
from ..core.clustering import MeanshiftClustering
from ..core.transforms import SignalTransformer
from ..utils.bandwidth import estimate_bandwidth

class TimeFrequencyVMD:
    """Implements Time-Frequency VMD with adaptive bandwidth and method selection."""
    
    def __init__(self, config: VMDConfig, bandwidth_config: BandwidthConfig):
        self.config = config
        self.bandwidth_config = bandwidth_config
        self.transformer = SignalTransformer(
            config.window, config.n_fft, config.modulated
        )
        self._dme_init = None  # Store DME result for IMR initialization
        
    def decompose(self, tf_map: np.ndarray, t_modes=[0,0.5,1]) -> ProcessingResult:
        """Main decomposition method that orchestrates the VMD process."""
        tf_map = np.ascontiguousarray(tf_map, dtype=np.complex128)
        C, F, N = tf_map.shape
        freqs = np.arange(1, F+1, dtype=np.float64)/F
        
        # Step 1: Calculate bandwidths for each time point
        bandwidths = self._calculate_bandwidths(tf_map, freqs)
        
        # Step 2: Determine decomposition method
        method = self._determine_method(tf_map, freqs, bandwidths)
        print(f"Selected decomposition method: {method}")
        
        # Step 3: Apply appropriate decomposition method
        clustering = MeanshiftClustering(self.bandwidth_config)
        if method == 'dme':
            result = self._apply_dme(tf_map, clustering, bandwidths, freqs)
        elif method == 'hvr':
            result = self._apply_hvr(tf_map, clustering, bandwidths, freqs)
        elif method == 'imr':
            result = self._apply_imr(tf_map, clustering, bandwidths, freqs)
        elif method == 'vmd':
            result = self._apply_vmd(tf_map, t_modes, freqs)
        result.method_used = method
        result.bandwidth_values = bandwidths
        
        return result

    def _calculate_bandwidths(self, tf_map: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Calculate adaptive bandwidths for each time point."""
        spectrum = np.abs(tf_map).mean(0).astype(np.float64)
        N = tf_map.shape[2]
        
        return np.array([
            estimate_bandwidth(
                freqs,
                spectrum[:,t].astype(np.float64),
                self.bandwidth_config,
                t,
                tf_map
            )
            for t in range(N)
        ])

    def _determine_method(self, tf_map: np.ndarray, freqs: np.ndarray, 
                         bandwidth: Union[float, np.ndarray]) -> str:
        """Determine whether to use DME or IMR based on signal characteristics."""
        if self.config.method != 'auto':
            return self.config.method
            
        # Analyze spectrum properties
        spectrum = np.abs(tf_map).mean(0)
        rho_min = np.min(spectrum)
        
        # Find minimum frequency separation between peaks
        peaks = scipy.signal.find_peaks(spectrum.mean(1))[0]
        delta_omega_min = (np.min(np.diff(freqs[peaks])) 
                          if len(peaks) >= 2 else np.inf)
        
        # Compare with bandwidth
        h = np.mean(bandwidth) if isinstance(bandwidth, np.ndarray) else bandwidth
        
        # Choose method based on criteria
        return 'dme' if (rho_min > self.config.rho_thresh and 
                         delta_omega_min > h) else 'imr'

    def _extract_dme_modes(self, tf_map: np.ndarray, clustering: MeanshiftClustering,
                           bandwidths: np.ndarray, freqs: np.ndarray,
                           keep_residual: bool = False) -> Dict:
        """Extract modes using DME method with option to keep or eliminate residuals."""
        C, F, N = tf_map.shape
        mode_functions = []
        time_indices = []
        center_frequencies = []
        
        # Store tf_map energy for validation
        tf_energy = np.sum(np.abs(tf_map)**2)
        
        for t in tqdm(range(N), desc="DME Processing"):
            freq_slice = tf_map[:,:,t]
            t_bandwidth = bandwidths[t]
            spectrum = (np.square(np.abs(freq_slice))).mean(0)
            
            # Get cluster centers using Meanshift++
            cluster_centers = clustering.cluster(
                freqs, 
                spectrum,
                t=t,
                tf_map=tf_map
            )
            
            if len(cluster_centers) == 0:
                # If no centers found, treat entire signal as one mode
                mode_functions.append(freq_slice)
                time_indices.append(t)
                center_frequencies.append(np.mean(freqs))
                continue
            
            # Sort centers by frequency for consistency
            cluster_centers = np.sort(cluster_centers)
            
            if keep_residual:
                # Original DME: Keep points within bandwidth, leave residual
                for center in cluster_centers:
                    distances = np.abs(freqs - center)
                    if np.isscalar(t_bandwidth):
                        cluster_mask = distances <= t_bandwidth
                    else:
                        cluster_mask = distances <= np.mean(t_bandwidth)
                    
                    mode = np.zeros_like(freq_slice)
                    mode[:,cluster_mask] = freq_slice[:,cluster_mask]
                    
                    mode_functions.append(mode)
                    time_indices.append(t)
                    center_frequencies.append(center)
            else:
                # Complete assignment: Every point goes to nearest center
                modes = [np.zeros_like(freq_slice) for _ in cluster_centers]
                
                for f in range(F):
                    # Find nearest center
                    distances = np.abs(cluster_centers - freqs[f])
                    nearest_center_idx = np.argmin(distances)
                    
                    # Assign the frequency component to the corresponding mode
                    modes[nearest_center_idx][:,f] = freq_slice[:,f]
                
                # Add modes to results
                for mode, center in zip(modes, cluster_centers):
                    mode_functions.append(mode)
                    time_indices.append(t)
                    center_frequencies.append(center)
        
        # Verify reconstruction and energy conservation
        reconstructed_tf_map = np.zeros_like(tf_map, dtype=complex)
        for t in range(N):
            t_modes = [mode for mode, tidx in zip(mode_functions, time_indices) if tidx == t]
            reconstructed_tf_map[:,:,t] = np.sum(t_modes, axis=0)
        
        recon_energy = np.sum(np.abs(reconstructed_tf_map)**2)
        energy_diff = np.abs(tf_energy - recon_energy) / tf_energy
        
        if energy_diff > 1e-10:
            print(f"Warning: DME energy conservation error: {energy_diff:.2e}")
            
        return {
            'mode_functions': mode_functions,
            'time_indices': time_indices,
            'center_frequencies': center_frequencies
        }

    def _apply_dme(self, tf_map: np.ndarray, clustering: MeanshiftClustering,
                    bandwidths: np.ndarray, freqs: np.ndarray) -> ProcessingResult:
        """Apply DME with configurable residual handling."""
        # Add keep_residual to config if not present
        if not hasattr(self.config, 'keep_residual'):
            self.config.keep_residual = False
            
        modes_data = self._extract_dme_modes(
            tf_map, 
            clustering, 
            bandwidths, 
            freqs,
            keep_residual=self.config.keep_residual
        )
        
        # Calculate residual only if keeping it
        if self.config.keep_residual:
            residual = self._calculate_residual(
                tf_map,
                modes_data['mode_functions'],
                modes_data['time_indices']
            )
        else:
            residual = np.zeros_like(tf_map, dtype=complex)
        
        return ProcessingResult(
            tf_map=tf_map,
            mode_functions=modes_data['mode_functions'],
            time_indices=modes_data['time_indices'],
            center_frequencies=modes_data['center_frequencies'],
            residual=residual,
            convergence_history=[],
            method_used='dme',
            bandwidth_values=bandwidths
        )

    def _check_convergence(self, max_diff: float, iteration: int) -> bool:
        """Check if convergence criteria are met with adaptive tolerance."""
        if iteration < 5:  # Minimum iterations to ensure stability
            return False
            
        # Adaptive tolerance based on iteration count
        adaptive_tol = self.config.tol * (1 + 5/iteration)  # Relaxed tolerance in early iterations
        
        # Check both absolute and relative convergence
        return (max_diff < adaptive_tol) or (
            iteration > 10 and max_diff < self.config.tol * 10  # Allow earlier stopping if close enough
        )

    def _extract_final_modes_with_energy(self, u_hat_plus: np.ndarray, omega_plus: list,
                                   K: np.ndarray, n: int, N: int, C: int,
                                   convergence_history: list, target_energy: float) -> Dict:
        """Extract and validate final modes with energy conservation."""
        mode_functions = []
        time_indices = []
        center_frequencies = []
        
        for t in range(N):
            if K[t] == 0:
                continue
                
            # Get valid modes for this timepoint
            t_modes = []
            t_freqs = []
            t_energies = []
            
            for k in range(K[t]):
                mode = u_hat_plus[n,:,:,k,t]
                freq = omega_plus[t][n,k]
                
                # Skip invalid modes
                if np.any(np.isnan(mode)) or np.any(np.isinf(mode)):
                    continue
                
                # Calculate mode energy
                mode_energy = np.sum(np.abs(mode)**2)
                if mode_energy < 1e-10 * target_energy:  # Energy threshold
                    continue
                    
                if not (0 <= freq <= 1):  # Frequency bounds check
                    continue
                    
                t_modes.append(mode)
                t_freqs.append(freq)
                t_energies.append(mode_energy)
            
            if t_modes:
                # Sort modes by frequency
                sort_idx = np.argsort(t_freqs)
                t_modes = [t_modes[i] for i in sort_idx]
                t_freqs = [t_freqs[i] for i in sort_idx]
                t_energies = [t_energies[i] for i in sort_idx]
                
                # Scale modes to preserve energy if needed
                total_t_energy = sum(t_energies)
                if total_t_energy > 0:
                    scale = np.sqrt(target_energy / N / total_t_energy)
                    t_modes = [mode * scale for mode in t_modes]
                
                # Store valid modes
                for mode, freq in zip(t_modes, t_freqs):
                    mode_functions.append(mode)
                    time_indices.append(t)
                    center_frequencies.append(freq)
        
        # If no valid modes found, fall back to DME initialization
        if not mode_functions:
            print("Warning: No valid IMR modes found, using DME initialization")
            return {
                'mode_functions': self._dme_init.mode_functions,
                'time_indices': self._dme_init.time_indices,
                'center_frequencies': self._dme_init.center_frequencies,
                'convergence_history': convergence_history
            }
        
        return {
            'mode_functions': mode_functions,
            'time_indices': time_indices,
            'center_frequencies': center_frequencies,
            'convergence_history': convergence_history
        }

    def _calculate_residual(self, tf_map: np.ndarray, 
                          mode_functions: List[np.ndarray],
                          time_indices: List[int]) -> np.ndarray:
        """Calculate residual signal from modes."""
        residual = np.zeros_like(tf_map, dtype=complex)
        
        for t in range(tf_map.shape[2]):
            t_modes = [mode_functions[i] for i, tidx in enumerate(time_indices) 
                      if tidx == t]
            if t_modes:
                reconstructed = np.sum(t_modes, axis=0)
                residual[:,:,t] = tf_map[:,:,t] - reconstructed
                
        return residual

    def generate_time_domain_modes(self, 
                                result: ProcessingResult,
                                transform_type: str = 'ssq_stft') -> ProcessingResult:
        """Generate time domain modes from time-frequency representation."""
        if not result.mode_functions:
            result.time_domain_modes = []
            return result
            
        time_domain_modes = self._process_modes_to_time_domain(
            result.mode_functions,
            result.time_indices,
            transform_type
        )
        
        result.time_domain_modes = time_domain_modes
        return result

    def _process_modes_to_time_domain(self, mode_functions: List[np.ndarray],
                                    time_indices: List[int],
                                    transform_type: str) -> List[np.ndarray]:
        """Process modes from time-frequency to time domain."""
        C, F = mode_functions[0].shape
        N = max(time_indices) + 1
        time_domain_modes = []
        
        for mode, t_idx in zip(mode_functions, time_indices):
            tf_map = np.zeros((C, F, N), dtype=complex)
            tf_map[:, :, t_idx] = mode
            
            mode_signal = self._transform_mode_to_time_domain(
                tf_map, C, transform_type
            )
            time_domain_modes.append(mode_signal)
            
        return time_domain_modes

    def _transform_mode_to_time_domain(self, tf_map: np.ndarray,
                                     C: int, transform_type: str) -> np.ndarray:
        """Transform a single mode to time domain."""
        mode_signal = np.zeros((C, tf_map.shape[2]), dtype=np.float32)
        
        for c in range(C):
            mode_signal[c] = self.transformer.inverse_transform(
                tf_map[c,:,:],
                transform_type=transform_type
            )
            
        return mode_signal

    def reconstruct_signal(self, result: ProcessingResult, 
                         transform_type: str = 'ssq_stft') -> np.ndarray:
        """Reconstruct time domain signal from modes."""
        if result.time_domain_modes is None:
            result = self.generate_time_domain_modes(result, transform_type)
        
        return np.sum(result.time_domain_modes, axis=0)

    def _prepare_bandwidth_array(self, bandwidth: Union[float, np.ndarray],
                               freqs: np.ndarray) -> np.ndarray:
        """Prepare bandwidth array ensuring proper shape and interpolation."""
        if np.isscalar(bandwidth):
            return np.full_like(freqs, bandwidth)
            
        if bandwidth.shape != freqs.shape:
            old_points = np.linspace(0, 1, len(bandwidth))
            f_interp = interp1d(old_points, bandwidth, kind='linear')
            return f_interp(freqs)
            
        return bandwidth

    def _update_single_timepoint(self, t: int, n: int, K_t: int,
                               tf_map: np.ndarray, u_hat_plus: np.ndarray,
                               sum_uk: np.ndarray, lambda_hat: np.ndarray,
                               omega_plus: list, freqs: np.ndarray,
                               bandwidth: Union[float, np.ndarray]) -> None:
        """Update modes and center frequencies for a single time point."""
        try:
            if K_t == 0:
                return  # Skip timepoints with no modes
                
            C = tf_map.shape[0]
            bandwidth_array = self._prepare_bandwidth_array(bandwidth, freqs)
            next_n = np.mod(n + 1, 2)
            
            # Process each mode sequentially
            for k in range(K_t):
                # Update sum_uk by removing current mode
                sum_uk[t] = sum_uk[t] - u_hat_plus[n,:,:,k,t]
                
                # More conservative alpha calculation for better stability
                other_freqs = np.array([omega_plus[t][n,i] for i in range(K_t) if i != k])
                if len(other_freqs) > 0:
                    min_sep = np.min(np.abs(omega_plus[t][n,k] - other_freqs))
                    local_alpha = self.config.alpha * np.minimum(1.0, min_sep / np.mean(bandwidth_array))
                else:
                    local_alpha = self.config.alpha
                
                # Calculate frequency weights with softer cutoff
                omega_k = omega_plus[t][n,k]
                freq_weights = 1.0 / (1.0 + 2*local_alpha*(freqs - omega_k)**2 + 1e-10)
                
                # Update mode with reduced momentum
                residual = tf_map[:,:,t] - sum_uk[t] - lambda_hat[n,:,:,t]/2
                new_mode = residual * freq_weights[:, None].T if freq_weights.ndim == 1 else residual * freq_weights
                momentum = 0.05  # Reduced momentum for better stability
                u_hat_plus[next_n,:,:,k,t] = (1 + momentum) * new_mode - momentum * u_hat_plus[n,:,:,k,t]
                
                # Update center frequency with reduced momentum
                if np.any(u_hat_plus[next_n,:,:,k,t]):
                    weights = np.abs(u_hat_plus[next_n,:,:,k,t])**2
                    weights_sum = np.sum(weights)
                    if weights_sum > 1e-10:
                        new_omega = np.sum(freqs * weights) / weights_sum
                        old_omega = omega_plus[t][n,k]
                        # Reduced momentum for frequency updates
                        omega_plus[t][next_n,k] = old_omega + 0.5 * (new_omega - old_omega)
                
                # Update sum_uk with new mode
                sum_uk[t] = sum_uk[t] + u_hat_plus[next_n,:,:,k,t]
                
        except Exception as e:
            print(f"Warning: Error updating timepoint {t}: {str(e)}")
            if t > 0:
                u_hat_plus[:,:,:,k,t] = u_hat_plus[:,:,:,k,t-1]
                omega_plus[t] = omega_plus[t-1].copy()

    def _apply_hvr(self, tf_map: np.ndarray, clustering: MeanshiftClustering,
                    bandwidths: np.ndarray, freqs: np.ndarray) -> ProcessingResult:
        """Hybrid Variational Refinement with DME-based initialization."""
        # Store original tf_map energy for validation
        tf_energy = np.sum(np.abs(tf_map)**2)
        
        # First use DME for initialization as per Algorithm 2 in Section II.E
        self._dme_init = self._apply_dme(tf_map, clustering, bandwidths, freqs)
        
        # Evaluate quality criteria using frequency separation (Equation 59) for each time point
        C, F, N = tf_map.shape
        refinement_mask = np.zeros(N, dtype=bool)  # Track which time points need refinement
        refinement_reasons = {}  # Store reasons for refinement decisions
        
        for t in range(N):
            t_modes = [i for i, tidx in enumerate(self._dme_init.time_indices) if tidx == t]
            needs_refinement = False
            reason = ""
            
            if len(t_modes) >= 2:
                centers = [self._dme_init.center_frequencies[i] for i in t_modes]
                # Calculate minimum frequency separation (Equation 59)
                delta_omega_sep = min(abs(centers[i] - centers[j]) 
                                    for i in range(len(centers)) 
                                    for j in range(i+1, len(centers)))
                
                # Get bandwidth value for this time point (handle both scalar and array cases)
                if np.isscalar(bandwidths):
                    bandwidth_t = bandwidths
                elif isinstance(bandwidths, np.ndarray):
                    if bandwidths.ndim == 0:  # 0-d array (scalar)
                        bandwidth_t = float(bandwidths)
                    elif bandwidths.size == 1:  # Single element array
                        bandwidth_t = float(bandwidths.flat[0])
                    elif t < len(bandwidths):
                        # Handle potential multi-dimensional bandwidth
                        bw_t = bandwidths[t]
                        if np.isscalar(bw_t):
                            bandwidth_t = float(bw_t)
                        elif hasattr(bw_t, '__len__') and len(bw_t) > 0:
                            bandwidth_t = float(np.mean(bw_t))  # Take mean if multi-dimensional
                        else:
                            bandwidth_t = float(bw_t)
                    else:
                        bandwidth_t = float(np.mean(bandwidths.flat))  # Use mean of all values
                else:
                    bandwidth_t = float(bandwidths)
                
                # Check if refinement is needed based on bandwidth comparison
                if delta_omega_sep <= bandwidth_t:
                    needs_refinement = True
                    reason = f"freq_sep={delta_omega_sep:.4f} <= bandwidth={bandwidth_t:.4f}"
                else:
                    reason = f"freq_sep={delta_omega_sep:.4f} > bandwidth={bandwidth_t:.4f}"
            elif len(t_modes) == 1:
                reason = "single mode - no refinement needed"
            else:
                reason = "no modes detected"
            
            refinement_mask[t] = needs_refinement
            refinement_reasons[t] = reason
        
        # Count time points requiring refinement
        refinement_count = np.sum(refinement_mask)
        total_points = N
        refinement_ratio = refinement_count / total_points
        
        print(f"Refinement analysis: {refinement_count}/{total_points} ({refinement_ratio:.1%}) time points need VMD refinement")
        
        # If no time points need refinement, return DME results
        if refinement_count == 0:
            print("DME quality sufficient for all time points - skipping VMD refinement")
            return ProcessingResult(
                tf_map=tf_map,
                mode_functions=self._dme_init.mode_functions,
                time_indices=self._dme_init.time_indices,
                center_frequencies=self._dme_init.center_frequencies,
                residual=self._dme_init.residual,
                convergence_history=[],
                method_used='hvr_dme_only',
                bandwidth_values=bandwidths
            )
        
        print(f"Applying VMD refinement to {refinement_count} time points with DME initialization")
        
        # Initialize VMD with DME results for improved convergence
        omega_plus = []
        K = np.zeros(N, dtype=np.int8)
        
        for t in range(N):
            t_modes = [i for i, tidx in enumerate(self._dme_init.time_indices) if tidx == t]
            K[t] = len(t_modes)
            if K[t] > 0:
                centers = [self._dme_init.center_frequencies[i] for i in t_modes]
                omega_plus.append(
                    np.repeat(
                        np.array(centers).reshape(1,-1),
                        2,
                        axis=0
                    )
                )
            else:
                # Fallback initialization if DME found no modes
                centers = clustering.cluster(freqs, np.abs(tf_map[:,:,t]).mean(0), t=t, tf_map=tf_map)
                if len(centers) == 0:
                    centers = [0.5]  # Default center if clustering fails
                K[t] = len(centers)
                omega_plus.append(
                    np.repeat(
                        np.array(centers).reshape(1,-1),
                        2,
                        axis=0
                    )
                )
        
        # Initialize arrays using DME modes for better starting point
        max_K = max(K)
        u_hat_plus = np.zeros([2, C, len(freqs), max_K, N], dtype=complex)
        sum_uk = np.zeros([N, C, len(freqs)], dtype=complex)
        lambda_hat = np.zeros([2, C, len(freqs), N], dtype=complex)
        
        # Initialize u_hat_plus with DME results for better convergence
        for t in range(N):
            t_modes = [i for i, tidx in enumerate(self._dme_init.time_indices) if tidx == t]
            for k, mode_idx in enumerate(t_modes):
                if k < max_K:  # Ensure we don't exceed array bounds
                    u_hat_plus[0,:,:,k,t] = self._dme_init.mode_functions[mode_idx]
                    u_hat_plus[1,:,:,k,t] = self._dme_init.mode_functions[mode_idx]
        
        convergence_history = []
        
        # Enhanced stopping criteria for practical VMD convergence
        convergence_window = 5  # Number of iterations to check for stagnation
        min_iterations = 3      # Minimum iterations before allowing convergence
        stagnation_threshold = 1e-10  # Threshold for detecting stagnation
        practical_tolerance = max(self.config.tol * 10, 1e-6)  # More practical tolerance
        
        # Create progress bar with HVR-specific description
        pbar = tqdm(range(self.config.max_iterations),
                desc="HVR Processing (VMD with DME init)",
                leave=True,
                position=0)
        
        try:
            for iteration in pbar:
                n = np.mod(iteration, 2)
                next_n = np.mod(iteration + 1, 2)
                
                # Apply standard ADMM iterations (Equations 4-6) with selective refinement
                for t in range(N):
                    if K[t] > 0:  # Only process if we have modes at this timepoint
                        # Check if this time point needs refinement
                        if refinement_mask[t]:
                            # Apply VMD updates for time points requiring refinement
                            # Reset sum_uk for this timepoint
                            sum_uk[t] = np.sum(u_hat_plus[n,:,:,:K[t],t], axis=2)
                            
                            try:
                                self._update_single_timepoint(
                                    t, n, K[t], tf_map, u_hat_plus, sum_uk, 
                                    lambda_hat, omega_plus, freqs, bandwidths[t]
                                )
                            except Exception as e:
                                pbar.write(f"Warning: Error at time {t}, iteration {iteration}: {str(e)}")
                                if t > 0:
                                    u_hat_plus[:,:,:,:K[t],t] = u_hat_plus[:,:,:,:K[t],t-1]
                                    omega_plus[t] = omega_plus[t-1].copy()
                        else:
                            # For time points that don't need refinement, keep DME results
                            # No updates needed - DME initialization is sufficient
                            pass
                
                # Update Lagrangian multipliers
                for t in range(N):
                    if K[t] > 0:
                        lambda_hat[next_n,:,:,t] = (
                            lambda_hat[n,:,:,t] + 
                            self.config.tau * (
                                np.sum(u_hat_plus[next_n,:,:,:K[t],t], axis=2) - 
                                tf_map[:,:,t]
                            )
                        )
                
                # Verify energy conservation
                total_recon = np.zeros_like(tf_map, dtype=complex)
                for t in range(N):
                    if K[t] > 0:
                        total_recon[:,:,t] = np.sum(u_hat_plus[next_n,:,:,:K[t],t], axis=2)
                
                current_energy = np.sum(np.abs(total_recon)**2)
                energy_diff = np.abs(tf_energy - current_energy) / tf_energy
                
                if energy_diff > 0.1:  # Energy conservation check
                    # Scale modes to preserve energy
                    scale_factor = np.sqrt(tf_energy / current_energy)
                    u_hat_plus[next_n] *= scale_factor
                
                # Enhanced convergence calculation with multiple criteria
                u_diff = np.zeros((C, N))
                freq_diff = 0.0
                
                for t in range(N):
                    if K[t] > 0:
                        for k in range(K[t]):
                            # Mode function convergence
                            mode_diff = np.mean(np.abs(
                                u_hat_plus[0,:,:,k,t] - u_hat_plus[1,:,:,k,t]
                            )**2)
                            u_diff[:,t] += mode_diff
                            
                            # Center frequency convergence
                            if len(omega_plus[t]) > n and len(omega_plus[t]) > next_n:
                                freq_change = abs(omega_plus[t][n,k] - omega_plus[t][next_n,k])
                                freq_diff = max(freq_diff, freq_change)
                        
                        u_diff[:,t] /= K[t]  # Average over modes
                
                max_mode_diff = np.max(u_diff)
                
                # Store convergence metrics
                convergence_metrics = {
                    'mode_diff': max_mode_diff,
                    'freq_diff': freq_diff,
                    'energy_diff': energy_diff,
                    'iteration': iteration
                }
                convergence_history.append([max_mode_diff, energy_diff, freq_diff])
                
                # Multi-criteria stopping conditions
                converged = False
                convergence_reason = ""
                
                if iteration >= min_iterations:
                    # Criterion 1: Standard VMD convergence (Equation 7)
                    if max_mode_diff < practical_tolerance:
                        converged = True
                        convergence_reason = "mode convergence"
                    
                    # Criterion 2: Frequency stability
                    elif freq_diff < 1e-6:
                        converged = True
                        convergence_reason = "frequency stability"
                    
                    # Criterion 3: Energy conservation with mode stability
                    elif energy_diff < 1e-6 and max_mode_diff < practical_tolerance * 100:
                        converged = True
                        convergence_reason = "energy-mode stability"
                    
                    # Criterion 4: Stagnation detection
                    elif len(convergence_history) >= convergence_window:
                        recent_diffs = [ch[0] for ch in convergence_history[-convergence_window:]]
                        diff_variance = np.var(recent_diffs)
                        if diff_variance < stagnation_threshold:
                            converged = True
                            convergence_reason = "stagnation detected"
                    
                    # Criterion 5: Practical early stopping for good enough results
                    elif (iteration >= 10 and max_mode_diff < practical_tolerance * 10 and 
                        energy_diff < 1e-4):
                        converged = True
                        convergence_reason = "practical threshold"
                
                # Update progress bar with enhanced metrics including refinement info
                pbar.set_postfix({
                    'mode_diff': f'{max_mode_diff:.2e}',
                    'freq_diff': f'{freq_diff:.2e}',
                    'energy_diff': f'{energy_diff:.2e}',
                    'modes': f'{max_K}',
                    'refine_pts': f'{refinement_count}',
                    'method': 'HVR'
                })
                
                # Check enhanced convergence criteria
                if converged:
                    pbar.write(f"HVR converged at iteration {iteration} ({convergence_reason})")
                    pbar.write(f"  Mode diff: {max_mode_diff:.2e}, Freq diff: {freq_diff:.2e}")
                    break
                    
                # Safety checks
                if np.isnan(max_mode_diff) or np.isnan(freq_diff):
                    pbar.write("Warning: NaN detected in HVR convergence calculation")
                    break
                
                # Forced termination for very long runs
                if iteration >= self.config.max_iterations - 1:
                    pbar.write(f"HVR terminated at max iterations ({self.config.max_iterations})")
                    pbar.write(f"  Final - Mode diff: {max_mode_diff:.2e}, Freq diff: {freq_diff:.2e}")
                    break
                    
        except Exception as e:
            pbar.write(f"Error in HVR iteration: {str(e)}")
            import traceback
            pbar.write(traceback.format_exc())
        finally:
            pbar.close()
        
        # Extract final refined modes with energy validation
        modes_data = self._extract_final_modes_with_energy(
            u_hat_plus, omega_plus, K, next_n, N, C,
            convergence_history, tf_energy
        )
        
        # Calculate final residual
        residual = self._calculate_residual(
            tf_map,
            modes_data['mode_functions'],
            modes_data['time_indices']
        )
        
        return ProcessingResult(
            tf_map=tf_map,
            mode_functions=modes_data['mode_functions'],
            time_indices=modes_data['time_indices'],
            center_frequencies=modes_data['center_frequencies'],
            residual=residual,
            convergence_history=modes_data['convergence_history'],
            method_used='hvr',
            bandwidth_values=bandwidths
        )
    
    def _apply_imr(self, tf_map: np.ndarray, clustering: MeanshiftClustering,
                  bandwidths: np.ndarray, freqs: np.ndarray) -> ProcessingResult:
        """Variational Mode Decomposition with Interaction Mode Resolution."""
        # Store original tf_map energy for validation
        tf_energy = np.sum(np.abs(tf_map)**2)
        
        # First use DME for initialization as per Section III.F of the paper
        self._dme_init = self._apply_dme(tf_map, clustering, bandwidths, freqs)
        
        # Initialize modes and frequencies using DME results
        C, F, N = tf_map.shape
        omega_plus = []
        K = np.zeros(N, dtype=np.int8)
        
        for t in range(N):
            t_modes = [i for i, tidx in enumerate(self._dme_init.time_indices) if tidx == t]
            K[t] = len(t_modes)
            if K[t] > 0:
                centers = [self._dme_init.center_frequencies[i] for i in t_modes]
                omega_plus.append(
                    np.repeat(
                        np.array(centers).reshape(1,-1),
                        2,
                        axis=0
                    )
                )
            else:
                # Fallback initialization if DME found no modes
                centers = clustering.cluster(freqs, np.abs(tf_map[:,:,t]).mean(0), t=t, tf_map=tf_map)
                if len(centers) == 0:
                    centers = [0.5]  # Default center if clustering fails
                K[t] = len(centers)
                omega_plus.append(
                    np.repeat(
                        np.array(centers).reshape(1,-1),
                        2,
                        axis=0
                    )
                )
        
        # Initialize arrays using DME modes
        max_K = max(K)
        u_hat_plus = np.zeros([2, C, len(freqs), max_K, N], dtype=complex)
        sum_uk = np.zeros([N, C, len(freqs)], dtype=complex)
        lambda_hat = np.zeros([2, C, len(freqs), N], dtype=complex)
        
        # Initialize u_hat_plus with DME results for better convergence
        for t in range(N):
            t_modes = [i for i, tidx in enumerate(self._dme_init.time_indices) if tidx == t]
            for k, mode_idx in enumerate(t_modes):
                if k < max_K:  # Ensure we don't exceed array bounds
                    u_hat_plus[0,:,:,k,t] = self._dme_init.mode_functions[mode_idx]
                    u_hat_plus[1,:,:,k,t] = self._dme_init.mode_functions[mode_idx]
        
        convergence_history = []
        
        # Enhanced stopping criteria for practical VMD convergence
        convergence_window = 5  # Number of iterations to check for stagnation
        min_iterations = 3      # Minimum iterations before allowing convergence
        stagnation_threshold = 1e-10  # Threshold for detecting stagnation
        practical_tolerance = max(self.config.tol * 10, 1e-6)  # More practical tolerance
        
        # Create progress bar with additional information
        pbar = tqdm(range(self.config.max_iterations),
                desc="IMR Processing",
                leave=True,
                position=0)
        
        try:
            for iteration in pbar:
                n = np.mod(iteration, 2)
                next_n = np.mod(iteration + 1, 2)
                
                # Update modes and frequencies for each time point
                for t in range(N):
                    if K[t] > 0:  # Only process if we have modes at this timepoint
                        # Reset sum_uk for this timepoint
                        sum_uk[t] = np.sum(u_hat_plus[n,:,:,:K[t],t], axis=2)
                        
                        try:
                            self._update_single_timepoint(
                                t, n, K[t], tf_map, u_hat_plus, sum_uk, 
                                lambda_hat, omega_plus, freqs, bandwidths[t]
                            )
                        except Exception as e:
                            pbar.write(f"Warning: Error at time {t}, iteration {iteration}: {str(e)}")
                            if t > 0:
                                u_hat_plus[:,:,:,:K[t],t] = u_hat_plus[:,:,:,:K[t],t-1]
                                omega_plus[t] = omega_plus[t-1].copy()
                
                # Update Lagrangian multipliers
                for t in range(N):
                    if K[t] > 0:
                        lambda_hat[next_n,:,:,t] = (
                            lambda_hat[n,:,:,t] + 
                            self.config.tau * (
                                np.sum(u_hat_plus[next_n,:,:,:K[t],t], axis=2) - 
                                tf_map[:,:,t]
                            )
                        )
                
                # Verify energy conservation
                total_recon = np.zeros_like(tf_map, dtype=complex)
                for t in range(N):
                    if K[t] > 0:
                        total_recon[:,:,t] = np.sum(u_hat_plus[next_n,:,:,:K[t],t], axis=2)
                
                current_energy = np.sum(np.abs(total_recon)**2)
                energy_diff = np.abs(tf_energy - current_energy) / tf_energy
                
                if energy_diff > 0.1:  # Energy conservation check
                    # Scale modes to preserve energy
                    scale_factor = np.sqrt(tf_energy / current_energy)
                    u_hat_plus[next_n] *= scale_factor
                
                # Enhanced convergence calculation with multiple criteria
                u_diff = np.zeros((C, N))
                freq_diff = 0.0
                
                for t in range(N):
                    if K[t] > 0:
                        for k in range(K[t]):
                            # Mode function convergence
                            mode_diff = np.mean(np.abs(
                                u_hat_plus[0,:,:,k,t] - u_hat_plus[1,:,:,k,t]
                            )**2)
                            u_diff[:,t] += mode_diff
                            
                            # Center frequency convergence
                            if len(omega_plus[t]) > n and len(omega_plus[t]) > next_n:
                                freq_change = abs(omega_plus[t][n,k] - omega_plus[t][next_n,k])
                                freq_diff = max(freq_diff, freq_change)
                        
                        u_diff[:,t] /= K[t]  # Average over modes
                
                max_mode_diff = np.max(u_diff)
                
                # Store convergence metrics
                convergence_history.append([max_mode_diff, energy_diff, freq_diff])
                
                # Multi-criteria stopping conditions
                converged = False
                convergence_reason = ""
                
                if iteration >= min_iterations:
                    # Criterion 1: Standard VMD convergence (Equation 7)
                    if max_mode_diff < practical_tolerance:
                        converged = True
                        convergence_reason = "mode convergence"
                    
                    # Criterion 2: Frequency stability
                    elif freq_diff < 1e-6:
                        converged = True
                        convergence_reason = "frequency stability"
                    
                    # Criterion 3: Energy conservation with mode stability
                    elif energy_diff < 1e-6 and max_mode_diff < practical_tolerance * 100:
                        converged = True
                        convergence_reason = "energy-mode stability"
                    
                    # Criterion 4: Stagnation detection
                    elif len(convergence_history) >= convergence_window:
                        recent_diffs = [ch[0] for ch in convergence_history[-convergence_window:]]
                        diff_variance = np.var(recent_diffs)
                        if diff_variance < stagnation_threshold:
                            converged = True
                            convergence_reason = "stagnation detected"
                    
                    # Criterion 5: Practical early stopping for good enough results
                    elif (iteration >= 10 and max_mode_diff < practical_tolerance * 10 and 
                        energy_diff < 1e-4):
                        converged = True
                        convergence_reason = "practical threshold"
                
                # Update progress bar with enhanced metrics
                pbar.set_postfix({
                    'mode_diff': f'{max_mode_diff:.2e}',
                    'freq_diff': f'{freq_diff:.2e}',
                    'energy_diff': f'{energy_diff:.2e}',
                    'modes': f'{max_K}',
                    'method': 'IMR'
                })
                
                # Check enhanced convergence criteria
                if converged:
                    pbar.write(f"IMR converged at iteration {iteration} ({convergence_reason})")
                    pbar.write(f"  Mode diff: {max_mode_diff:.2e}, Freq diff: {freq_diff:.2e}")
                    break
                    
                # Safety checks
                if np.isnan(max_mode_diff) or np.isnan(freq_diff):
                    pbar.write("Warning: NaN detected in IMR convergence calculation")
                    break
                
                # Forced termination for very long runs
                if iteration >= self.config.max_iterations - 1:
                    pbar.write(f"IMR terminated at max iterations ({self.config.max_iterations})")
                    pbar.write(f"  Final - Mode diff: {max_mode_diff:.2e}, Freq diff: {freq_diff:.2e}")
                    break
                    
        except Exception as e:
            pbar.write(f"Error in IMR iteration: {str(e)}")
            import traceback
            pbar.write(traceback.format_exc())
        finally:
            pbar.close()
        
        # Extract final modes with energy validation
        modes_data = self._extract_final_modes_with_energy(
            u_hat_plus, omega_plus, K, next_n, N, C,
            convergence_history, tf_energy
        )
        
        # Calculate final residual
        residual = self._calculate_residual(
            tf_map,
            modes_data['mode_functions'],
            modes_data['time_indices']
        )
        
        return ProcessingResult(
            tf_map=tf_map,
            mode_functions=modes_data['mode_functions'],
            time_indices=modes_data['time_indices'],
            center_frequencies=modes_data['center_frequencies'],
            residual=residual,
            convergence_history=modes_data['convergence_history'],
            method_used='imr',
            bandwidth_values=bandwidths
        )

    def _apply_vmd(self, tf_map: np.ndarray, t_modes: np.ndarray, freqs: np.ndarray) -> ProcessingResult:
        """Variational Mode Decomposition."""
        # Store original tf_map energy for validation
        tf_energy = np.sum(np.abs(tf_map)**2)

        # Initialize modes and frequencies using traditional VMD initialization
        C, F, N = tf_map.shape
        omega_plus = []
        K = np.zeros(N, dtype=np.int8)
        
        for t in range(N):
            K[t] = len(t_modes)
            if K[t] > 0:
                centers = [t_modes[i] for i in range(len(t_modes))]
                omega_plus.append(
                    np.repeat(
                        np.array(centers).reshape(1,-1),
                        2,
                        axis=0
                    )
                )
        bandwidths = 1/len(t_modes)
        
        # Initialize arrays using traditional VMD initialization
        max_K = max(K)
        u_hat_plus = np.zeros([2, C, len(freqs), max_K, N], dtype=complex)
        sum_uk = np.zeros([N, C, len(freqs)], dtype=complex)
        lambda_hat = np.zeros([2, C, len(freqs), N], dtype=complex)
        
        convergence_history = []
        
        # Enhanced stopping criteria for practical VMD convergence
        convergence_window = 5  # Number of iterations to check for stagnation
        min_iterations = 3      # Minimum iterations before allowing convergence
        stagnation_threshold = 1e-10  # Threshold for detecting stagnation
        practical_tolerance = max(self.config.tol * 10, 1e-6)  # More practical tolerance
        
        # Create progress bar with additional information
        pbar = tqdm(range(self.config.max_iterations),
                desc="VMD Processing",
                leave=True,
                position=0)
        
        try:
            for iteration in pbar:
                n = np.mod(iteration, 2)
                next_n = np.mod(iteration + 1, 2)
                
                # Update modes and frequencies for each time point
                for t in range(N):
                    if K[t] > 0:  # Only process if we have modes at this timepoint
                        # Reset sum_uk for this timepoint
                        sum_uk[t] = np.sum(u_hat_plus[n,:,:,:K[t],t], axis=2)
                        
                        try:
                            self._update_single_timepoint(
                                t, n, K[t], tf_map, u_hat_plus, sum_uk, 
                                lambda_hat, omega_plus, freqs, bandwidths
                            )
                        except Exception as e:
                            pbar.write(f"Warning: Error at time {t}, iteration {iteration}: {str(e)}")
                            if t > 0:
                                u_hat_plus[:,:,:,:K[t],t] = u_hat_plus[:,:,:,:K[t],t-1]
                                omega_plus[t] = omega_plus[t-1].copy()
                
                # Update Lagrangian multipliers
                for t in range(N):
                    if K[t] > 0:
                        lambda_hat[next_n,:,:,t] = (
                            lambda_hat[n,:,:,t] + 
                            self.config.tau * (
                                np.sum(u_hat_plus[next_n,:,:,:K[t],t], axis=2) - 
                                tf_map[:,:,t]
                            )
                        )
                
                # Verify energy conservation
                total_recon = np.zeros_like(tf_map, dtype=complex)
                for t in range(N):
                    if K[t] > 0:
                        total_recon[:,:,t] = np.sum(u_hat_plus[next_n,:,:,:K[t],t], axis=2)
                
                current_energy = np.sum(np.abs(total_recon)**2)
                energy_diff = np.abs(tf_energy - current_energy) / tf_energy
                
                if energy_diff > 0.1:  # Energy conservation check
                    # Scale modes to preserve energy
                    scale_factor = np.sqrt(tf_energy / current_energy)
                    u_hat_plus[next_n] *= scale_factor
                
                # Enhanced convergence calculation with multiple criteria
                u_diff = np.zeros((C, N))
                freq_diff = 0.0
                
                for t in range(N):
                    if K[t] > 0:
                        for k in range(K[t]):
                            # Mode function convergence
                            mode_diff = np.mean(np.abs(
                                u_hat_plus[0,:,:,k,t] - u_hat_plus[1,:,:,k,t]
                            )**2)
                            u_diff[:,t] += mode_diff
                            
                            # Center frequency convergence
                            if len(omega_plus[t]) > n and len(omega_plus[t]) > next_n:
                                freq_change = abs(omega_plus[t][n,k] - omega_plus[t][next_n,k])
                                freq_diff = max(freq_diff, freq_change)
                        
                        u_diff[:,t] /= K[t]  # Average over modes
                
                max_mode_diff = np.max(u_diff)
                
                # Store convergence metrics
                convergence_history.append([max_mode_diff, energy_diff, freq_diff])
                
                # Multi-criteria stopping conditions
                converged = False
                convergence_reason = ""
                
                if iteration >= min_iterations:
                    # Criterion 1: Standard VMD convergence (Equation 7)
                    if max_mode_diff < practical_tolerance:
                        converged = True
                        convergence_reason = "mode convergence"
                    
                    # Criterion 2: Frequency stability
                    elif freq_diff < 1e-6:
                        converged = True
                        convergence_reason = "frequency stability"
                    
                    # Criterion 3: Energy conservation with mode stability
                    elif energy_diff < 1e-6 and max_mode_diff < practical_tolerance * 100:
                        converged = True
                        convergence_reason = "energy-mode stability"
                    
                    # Criterion 4: Stagnation detection
                    elif len(convergence_history) >= convergence_window:
                        recent_diffs = [ch[0] for ch in convergence_history[-convergence_window:]]
                        diff_variance = np.var(recent_diffs)
                        if diff_variance < stagnation_threshold:
                            converged = True
                            convergence_reason = "stagnation detected"
                    
                    # Criterion 5: Practical early stopping for good enough results
                    elif (iteration >= 10 and max_mode_diff < practical_tolerance * 10 and 
                        energy_diff < 1e-4):
                        converged = True
                        convergence_reason = "practical threshold"
                
                # Update progress bar with enhanced metrics
                pbar.set_postfix({
                    'mode_diff': f'{max_mode_diff:.2e}',
                    'freq_diff': f'{freq_diff:.2e}',
                    'energy_diff': f'{energy_diff:.2e}',
                    'modes': f'{max_K}',
                    'method': 'VMD'
                })
                
                # Check enhanced convergence criteria
                if converged:
                    pbar.write(f"VMD converged at iteration {iteration} ({convergence_reason})")
                    pbar.write(f"  Mode diff: {max_mode_diff:.2e}, Freq diff: {freq_diff:.2e}")
                    break
                    
                # Safety checks
                if np.isnan(max_mode_diff) or np.isnan(freq_diff):
                    pbar.write("Warning: NaN detected in VMD convergence calculation")
                    break
                
                # Forced termination for very long runs
                if iteration >= self.config.max_iterations - 1:
                    pbar.write(f"VMD terminated at max iterations ({self.config.max_iterations})")
                    pbar.write(f"  Final - Mode diff: {max_mode_diff:.2e}, Freq diff: {freq_diff:.2e}")
                    break
                    
        except Exception as e:
            pbar.write(f"Error in VMD iteration: {str(e)}")
            import traceback
            pbar.write(traceback.format_exc())
        finally:
            pbar.close()
        
        # Extract final modes with energy validation
        modes_data = self._extract_final_modes_with_energy(
            u_hat_plus, omega_plus, K, next_n, N, C,
            convergence_history, tf_energy
        )
        
        # Calculate final residual
        residual = self._calculate_residual(
            tf_map,
            modes_data['mode_functions'],
            modes_data['time_indices']
        )
        
        return ProcessingResult(
            tf_map=tf_map,
            mode_functions=modes_data['mode_functions'],
            time_indices=modes_data['time_indices'],
            center_frequencies=modes_data['center_frequencies'],
            residual=residual,
            convergence_history=modes_data['convergence_history'],
            method_used='vmd',
            bandwidth_values=bandwidths
        )