# tfvmd/core/transforms.py

from typing import Tuple, Optional
import numpy as np
from ssqueezepy import (stft, istft, cwt, icwt, ssq_stft, 
                       issq_stft, ssq_cwt, issq_cwt)

class SignalTransformer:
    """Handles signal transformations between time and time-frequency domains."""
    
    def __init__(self, window: np.ndarray, n_fft: int, modulated: bool = True):
        """Initialize transformer with window and FFT parameters."""
        self.window = window
        self.n_fft = n_fft
        self.modulated = modulated
        
    def transform(self, signal: np.ndarray, 
                 transform_type: str = 'ssq_stft') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform signal to time-frequency representation with consistent length handling.
        
        Args:
            signal: Input signal
            transform_type: Type of transform to apply
            
        Returns:
            Tuple of (transformed signal, optional scale information)
        """
        if signal.ndim == 1:
            signal = signal.reshape(1,-1)

        if transform_type == 'stft':
            tf_map = stft(signal, window=self.window, n_fft=self.n_fft,
                         hop_len=1, fs=1, padtype='reflect',
                         modulated=self.modulated, derivative=False)
            return tf_map, None
                       
        elif transform_type == 'ssq_stft':
            tf_map, *_ = ssq_stft(signal, window=self.window,
                                 n_fft=self.n_fft, hop_len=1,
                                 fs=1, padtype='reflect',
                                 modulated=self.modulated)
            return tf_map, None
            
        elif transform_type == 'cwt':
            tf_map, scale = cwt(signal, wavelet='gmw', 
                              scales='log-piecewise',
                              fs=1, padtype='reflect', 
                              rpadded=False, l1_norm=True)
            
            # Trim to original length
            # tf_map = tf_map[:, :self.original_length]
            return tf_map[:,::-1,:], scale
            
        elif transform_type == 'ssq_cwt':
            tf_map, _, _, scale = ssq_cwt(signal, wavelet='gmw',
                                        scales='log-piecewise')
            # tf_map = tf_map[:, :self.original_length]
            return tf_map[:,::-1,:], scale
            
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def inverse_transform(self, tf_map: np.ndarray, 
                         transform_type: str = 'ssq_stft') -> np.ndarray:
        """
        Inverse transform from time-frequency to time domain with length matching.
        
        Args:
            tf_map: Time-frequency representation
            transform_type: Type of transform to invert
            
        Returns:
            Time domain signal with original length
        """
        # Pad tf_map to account for transform effects
        
        if transform_type == 'stft':
            signal = istft(tf_map, window=self.window, n_fft=self.n_fft,
                         hop_len=1, modulated=self.modulated)
            
        elif transform_type == 'ssq_stft':
            signal = issq_stft(tf_map, window=self.window, n_fft=self.n_fft,
                             hop_len=1, modulated=self.modulated)
                           
        elif transform_type == 'cwt':
            # Reverse frequency axis back for CWT
            tf_map = tf_map[::-1,:]
            signal = icwt(tf_map, wavelet='gmw', scales='log-piecewise',
                         padtype='reflect', rpadded=False, l1_norm=True)
                       
        elif transform_type == 'ssq_cwt':
            tf_map = tf_map[::-1,]
            signal = issq_cwt(tf_map, wavelet='gmw')
            
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Extract center portion to match original length

        return signal