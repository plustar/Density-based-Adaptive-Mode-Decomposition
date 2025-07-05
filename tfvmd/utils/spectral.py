# tfvmd/utils/spectral.py

import numpy as np
from scipy import signal
from typing import Tuple, Union, Optional, List
from scipy.interpolate import interp1d

def compute_spectrogram(signal: np.ndarray,
                       fs: float,
                       window: Optional[np.ndarray] = None,
                       nperseg: Optional[int] = None,
                       noverlap: Optional[int] = None,
                       scaling: str = 'density') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spectrogram of a signal with enhanced parameters.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        window: Window function (default: Hann window)
        nperseg: Length of each segment
        noverlap: Number of points to overlap between segments
        scaling: Scaling type ('density' or 'spectrum')
        
    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    if nperseg is None:
        nperseg = min(256, len(signal))
    
    if window is None:
        window = signal.windows.hann(nperseg)
        
    if noverlap is None:
        noverlap = nperseg // 2
        
    freqs, times, Sxx = signal.spectrogram(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
        mode='complex'
    )
    
    return freqs, times, Sxx

def estimate_instantaneous_frequency(signal: np.ndarray,
                                  fs: float,
                                  method: str = 'hilbert',
                                  smooth_window: int = 5) -> np.ndarray:
    """
    Estimate instantaneous frequency using various methods.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        method: Method to use ('hilbert' or 'phase_diff')
        smooth_window: Window size for smoothing
        
    Returns:
        Array of instantaneous frequencies
    """
    if method == 'hilbert':
        analytic_signal = signal.hilbert(signal)
        phase = np.unwrap(np.angle(analytic_signal))
        inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
        # Add endpoint to match original signal length
        inst_freq = np.append(inst_freq, inst_freq[-1])
        
    elif method == 'phase_diff':
        phase = np.arctan2(
            signal.hilbert(signal).imag,
            signal
        )
        unwrapped_phase = np.unwrap(phase)
        inst_freq = np.gradient(unwrapped_phase) * fs / (2.0 * np.pi)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply smoothing
    if smooth_window > 1:
        window = np.ones(smooth_window) / smooth_window
        inst_freq = np.convolve(inst_freq, window, mode='same')
        
    return inst_freq

def ridge_detection(tf_map: np.ndarray,
                   freqs: np.ndarray,
                   threshold: float = 0.1,
                   min_length: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Detect ridges in time-frequency representation.
    
    Args:
        tf_map: Time-frequency representation
        freqs: Frequency points
        threshold: Power threshold for ridge detection
        min_length: Minimum ridge length to keep
        
    Returns:
        List of (times, frequencies) tuples for each ridge
    """
    power = np.abs(tf_map)**2
    mask = power > (threshold * np.max(power))
    
    ridges = []
    n_times = tf_map.shape[1]
    
    for t in range(n_times):
        peaks = signal.find_peaks(power[:,t])[0]
        valid_peaks = peaks[power[peaks,t] > threshold * np.max(power[:,t])]
        
        if len(valid_peaks) > 0:
            for peak in valid_peaks:
                # Check if peak connects to existing ridge
                connected = False
                for ridge in ridges:
                    if len(ridge[0]) > 0 and ridge[0][-1] == t-1:
                        last_freq_idx = np.searchsorted(freqs, ridge[1][-1])
                        if abs(last_freq_idx - peak) <= 2:  # Allow small frequency jumps
                            ridge[0] = np.append(ridge[0], t)
                            ridge[1] = np.append(ridge[1], freqs[peak])
                            connected = True
                            break
                
                if not connected:
                    # Start new ridge
                    ridges.append((np.array([t]), np.array([freqs[peak]])))
    
    # Filter short ridges
    long_ridges = [
        (times, frequencies) for times, frequencies in ridges
        if len(times) >= min_length
    ]
    
    return long_ridges

def adaptive_reassignment(tf_map: np.ndarray,
                         times: np.ndarray,
                         freqs: np.ndarray,
                         threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform adaptive time-frequency reassignment.
    
    Args:
        tf_map: Time-frequency representation
        times: Time points
        freqs: Frequency points
        threshold: Energy threshold for reassignment
        
    Returns:
        Tuple of (reassigned times, reassigned frequencies, reassigned energies)
    """
    dt = times[1] - times[0]
    df = freqs[1] - freqs[0]
    
    # Compute time and frequency gradients
    t_grad = np.gradient(tf_map, dt, axis=1)
    f_grad = np.gradient(tf_map, df, axis=0)
    
    # Compute reassignment operators
    energy = np.abs(tf_map)**2
    mask = energy > (threshold * np.max(energy))
    
    t_reassign = np.zeros_like(tf_map)
    f_reassign = np.zeros_like(tf_map)
    
    # Compute reassignment coordinates
    with np.errstate(divide='ignore', invalid='ignore'):
        t_reassign[mask] = -np.real(
            1j * t_grad[mask] / tf_map[mask]
        )
        f_reassign[mask] = -np.real(
            1j * f_grad[mask] / tf_map[mask]
        )
    
    # Limit reassignment distance
    max_shift = 5
    t_reassign = np.clip(t_reassign, -max_shift*dt, max_shift*dt)
    f_reassign = np.clip(f_reassign, -max_shift*df, max_shift*df)
    
    # Create output arrays
    reassigned_times = times.reshape(1,-1) + t_reassign
    reassigned_freqs = freqs.reshape(-1,1) + f_reassign
    reassigned_energy = energy * mask
    
    return reassigned_times, reassigned_freqs, reassigned_energy

def compute_synchrosqueezed_transform(signal: np.ndarray,
                                    fs: float,
                                    n_freqs: int = 256,
                                    gamma: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute synchrosqueezed transform of a signal.
    
    Args:
        signal: Input signal
        fs: Sampling frequency
        n_freqs: Number of frequency bins
        gamma: Threshold for synchrosqueezing
        
    Returns:
        Tuple of (frequencies, synchrosqueezed transform)
    """
    # Compute analytic signal
    analytic = signal.hilbert(signal)
    
    # Compute STFT
    window = signal.windows.hann(n_freqs)
    freqs, times, Zxx = signal.stft(
        analytic,
        fs=fs,
        window=window,
        nperseg=n_freqs,
        noverlap=n_freqs-1,
        boundary=None
    )
    
    # Compute frequency transform
    dt = 1/fs
    dw = 2*np.pi*fs/n_freqs
    
    # Estimate instantaneous frequency
    inst_freq = np.zeros_like(Zxx, dtype=float)
    mask = np.abs(Zxx) > gamma * np.max(np.abs(Zxx))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        inst_freq[mask] = np.real(
            1j * np.gradient(Zxx, dt, axis=1)[mask] / Zxx[mask]
        ) / (2*np.pi)
    
    # Perform synchrosqueezing
    Ts = np.zeros_like(Zxx, dtype=complex)
    
    for i, f in enumerate(freqs):
        if f == 0:
            continue
            
        # Find closest frequency bin
        idx = np.round((inst_freq[i,:] - f) / dw).astype(int)
        valid = (idx >= 0) & (idx < n_freqs) & mask[i,:]
        
        for j, k in enumerate(idx[valid]):
            Ts[k,j] += Zxx[i,j]
    
    return freqs, Ts