# tfvmd/visualization/plotter.py

from ..core.config import ProcessingResult
from .config import VisualizationConfig, SaveConfig

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional

class VMDVisualizer:
    """VMD visualization with three-panel layout."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Configure matplotlib style with scaled fonts."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Get scaled font sizes
        sizes = self.config.font_sizes
        
        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.size': sizes['font_size'],
            'axes.titlesize': sizes['title_size'],
            'axes.labelsize': sizes['label_size'],
            'xtick.labelsize': sizes['tick_size'],
            'ytick.labelsize': sizes['tick_size'],
            'legend.fontsize': sizes['legend_size'],
            'figure.dpi': self.config.dpi
        })

    def plot_analysis(self, 
                     result: ProcessingResult,
                     original_signal: np.ndarray,
                     fs: float,
                     figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """
        Create three-panel visualization.
        
        Args:
            result: ProcessingResult containing decomposition data
            original_signal: Original input signal
            fs: Sampling frequency
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, height_ratios=self.config.height_ratios)
        
        # Original time-frequency map
        ax_tf_orig = fig.add_subplot(gs[0, 0])
        self._plot_tf_original(ax_tf_orig, result, fs)
        
        # Center frequencies plot
        ax_tf_centers = fig.add_subplot(gs[0, 1])
        self._plot_tf_centers(ax_tf_centers, result, fs)
        
        # Signal comparison (full width)
        ax_signal = fig.add_subplot(gs[1, :])
        self._plot_signal_comparison(ax_signal, original_signal, result, fs)
        
        # Adjust layout
        plt.tight_layout()
        return fig
        
    def _plot_tf_reconstruct(self,
                         ax: Axes,
                         result: ProcessingResult,
                         fs: float,
                         title=None) -> None:
        """Plot original time-frequency map."""
        tf_energy = result.compute_tf_energy()
        time_points = np.arange(tf_energy.shape[1]) / fs
        freq_points = np.linspace(0, fs/2, tf_energy.shape[0])
        
        im = ax.pcolormesh(time_points, freq_points, tf_energy,
                          cmap=self.config.color_palette, 
                          shading='gouraud')
        
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        # if self.config.tf_legend.location:
        #     ax.legend(**self.config.tf_legend.get_legend_kwargs())
        plt.colorbar(im, ax=ax, label='Energy')
    
    def _plot_tf_original(self,
                         ax: Axes,
                         result: ProcessingResult,
                         fs: float,
                         title=None) -> None:
        """Plot original time-frequency map."""
        tf_energy = np.abs(result.tf_map)[0]
        time_points = np.arange(tf_energy.shape[1]) / fs
        freq_points = np.linspace(0, fs/2, tf_energy.shape[0])
        
        im = ax.pcolormesh(time_points, freq_points, tf_energy,
                          cmap=self.config.color_palette, 
                          shading='gouraud')
        
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        # if self.config.tf_legend.location:
        #     ax.legend(**self.config.tf_legend.get_legend_kwargs())
        plt.colorbar(im, ax=ax, label='Energy')
        
    def _plot_tf_centers(self,
                        ax: Axes,
                        result: ProcessingResult,
                        fs: float,
                        title=None) -> None:
        """Plot center frequencies with energy-based markers."""
        # Calculate mode energies for marker sizing
        mode_energies = self._calculate_mode_energies(result.mode_functions)
        normalized_energies = mode_energies / np.max(mode_energies)
        
        # Get time-frequency coordinates
        times = np.array(result.time_indices) / fs
        freqs = np.array(result.center_frequencies) * fs/2
        
        # Create empty background matching the original TF plot
        tf_energy = result.compute_tf_energy()
        time_points = np.arange(tf_energy.shape[1]) / fs
        freq_points = np.linspace(0, fs/2, tf_energy.shape[0])
        ax.pcolormesh(time_points, freq_points, np.zeros_like(tf_energy),
                     cmap='Greys', alpha=0.1)
        
        # Plot center frequencies
        scatter_sizes = normalized_energies * self.config.marker_base_size
        # ax.scatter(times, freqs, 
        #           s=scatter_sizes,
        #           c=self.config.marker_color,
        #           alpha=self.config.marker_alpha,
        #           edgecolors=self.config.marker_edge_color,
        #           linewidth=0.5,
        #           label='Center Frequencies')
        ax.scatter(times, freqs, 
                  s=scatter_sizes,
                  c=self.config.marker_color,
                  alpha=self.config.marker_alpha,
                  edgecolors=self.config.marker_edge_color,
                  linewidth=0.5)
        
        if title is not None:
            ax.set_title(title)
        # ax.set_title('Mode Center Frequencies')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        # ax.legend()
        
        # Match limits with original TF plot
        ax.set_xlim(time_points[0], time_points[-1])
        ax.set_ylim(freq_points[0], freq_points[-1])
        
    def _plot_signal_comparison(self,
                              ax: Axes,
                              original_signal: np.ndarray,
                              result: ProcessingResult,
                              fs: float) -> None:
        """Plot complete signal comparison."""
        time = np.arange(len(original_signal)) / fs
        
        # Plot original signal
        ax.plot(time, original_signal, '-', 
                color=self.config.original_color,
                alpha=self.config.original_alpha,
                label='Original', 
                linewidth=1)
        
        # Plot reconstructed signal
        if result.time_domain_modes:
            reconstruction = np.sum(result.time_domain_modes, axis=0)[0]
            ax.plot(time, reconstruction,
                   self.config.reconstruction_marker,
                   color=self.config.reconstruction_color,
                   markersize=self.config.reconstruction_size,
                   alpha=self.config.reconstruction_alpha,
                   label='Reconstructed')
            
            # Calculate and display reconstruction error
            mse = np.mean((original_signal - reconstruction)**2)
            ax.text(0.02, 0.98, f'MSE: {mse:.2e}',
                   transform=ax.transAxes,
                   verticalalignment='bottom',
                   bbox=dict(facecolor='white', alpha=0.2))
        
        ax.set_title('Signal Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        if self.config.signal_legend.location:
            ax.legend(**self.config.signal_legend.get_legend_kwargs())
        
    def _calculate_mode_energies(self, modes: list) -> np.ndarray:
        """Calculate energy for each mode."""
        energies = []
        for mode in modes:
            if isinstance(mode, np.ndarray):
                energy = np.mean(np.abs(mode)**2)
                energies.append(energy)
            else:
                energies.append(0)
        return np.array(energies)
    
    def save_figure(self, 
                   fig: Figure,
                   filename: str,
                   save_config: Optional[SaveConfig] = None) -> None:
        """
        Save figure in specified format.
        
        Args:
            fig: matplotlib Figure to save
            filename: Output filename (without extension)
            save_config: Optional SaveConfig to override default settings
        """
        config = save_config or self.config.save_config
        
        # Add extension if not present
        if not filename.endswith(f'.{config.format}'):
            filename = f'{filename}.{config.format}'
        
        # Save figure
        try:
            fig.savefig(filename, **config.get_save_kwargs())
            print(f"Figure saved successfully as {filename}")
        except Exception as e:
            print(f"Error saving figure: {str(e)}")
    
    def plot_and_save(self,
                      result: ProcessingResult,
                      original_signal: np.ndarray,
                      fs: float,
                      filename: str,
                      figsize: Tuple[int, int] = (12, 8),
                      save_config: Optional[SaveConfig] = None) -> Figure:
        """
        Create visualization and save to file.
        
        Args:
            result: ProcessingResult containing decomposition data
            original_signal: Original input signal
            fs: Sampling frequency
            filename: Output filename
            figsize: Figure size
            save_config: Optional SaveConfig to override default settings
            
        Returns:
            matplotlib Figure
        """
        # Create figure
        fig = self.plot_analysis(result, original_signal, fs, figsize)
        
        # Save figure
        self.save_figure(fig, filename, save_config)
        
        return fig