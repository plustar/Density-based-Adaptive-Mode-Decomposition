# tfvmd/visualization/config.py

from dataclasses import dataclass, field
from typing import Tuple, Dict, Union, Optional

@dataclass
class FontConfig:
    """Base font sizes that will be scaled."""
    base_font: int = 10
    base_title: int = 11 
    base_label: int = 10
    base_tick: int = 8
    base_legend: int = 8
    
    def get_scaled_sizes(self, scale: float) -> Dict[str, int]:
        """Get all font sizes scaled by the given factor."""
        return {
            'font_size': int(self.base_font * scale),
            'title_size': int(self.base_title * scale),
            'label_size': int(self.base_label * scale),
            'tick_size': int(self.base_tick * scale),
            'legend_size': int(self.base_legend * scale)
        }

@dataclass
class SaveConfig:
    """Configuration for figure saving."""
    format: str = 'svg'  # 'svg', 'eps', 'pdf', 'png'
    dpi: int = 300  # DPI for raster formats
    bbox_inches: str = 'tight'  # 'tight' or None
    transparent: bool = False  # Transparent background
    pad_inches: float = 0.1  # Padding around figure
    
    def get_save_kwargs(self) -> Dict:
        """Get dictionary of save parameters."""
        kwargs = {
            'format': self.format,
            'bbox_inches': self.bbox_inches,
            'transparent': self.transparent,
            'pad_inches': self.pad_inches
        }
        if self.format not in ['svg', 'eps', 'pdf']:  # Raster formats
            kwargs['dpi'] = self.dpi
        return kwargs

@dataclass
class LegendConfig:
    """Configuration for legend positioning and style."""
    # Location can be:
    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 
    # 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    # or a tuple of (x, y) coordinates in axes coordinates (0,0 is bottom left, 1,1 is top right)
    location: Union[str, Tuple[float, float]] = 'best'
    
    # Legend box placement relative to the point specified by loc
    bbox_to_anchor: Optional[Tuple[float, float]] = None
    
    # Frame style
    frameon: bool = True  # Whether to draw the frame
    framealpha: float = 0.8  # Frame transparency
    
    # Layout
    ncol: int = 1  # Number of columns
    
    # Margins and padding
    borderpad: float = 0.4  # Padding between legend and edge
    labelspacing: float = 0.5  # Vertical space between entries
    handletextpad: float = 0.8  # Space between marker and text
    
    def get_legend_kwargs(self) -> Dict:
        """Get dictionary of legend parameters."""
        kwargs = {
            'loc': self.location,
            'frameon': self.frameon,
            'framealpha': self.framealpha,
            'ncol': self.ncol,
            'borderpad': self.borderpad,
            'labelspacing': self.labelspacing,
            'handletextpad': self.handletextpad
        }
        if self.bbox_to_anchor is not None:
            kwargs['bbox_to_anchor'] = self.bbox_to_anchor
        return kwargs

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    # Font settings
    font_family: str = 'Times New Roman'
    font_scale: float = 1.0  # Scale factor for all fonts
    base_fonts: FontConfig = field(default_factory=FontConfig)
    
    # Display settings
    dpi: int = 100
    color_palette: str = 'viridis'
    
    # Frequency marker settings
    marker_base_size: float = 100  
    marker_alpha: float = 0.6
    marker_color: str = 'red'
    marker_edge_color: str = 'white'
    
    # Signal comparison settings
    original_color: str = 'blue'
    original_alpha: float = 0.8
    reconstruction_color: str = 'red'
    reconstruction_alpha: float = 0.7
    reconstruction_marker: str = '*'
    reconstruction_size: float = 4
    
    # Layout settings
    height_ratios: Tuple[float, float] = (1, 0.8)
    
    @property
    def font_sizes(self) -> Dict[str, int]:
        """Get all font sizes scaled by font_scale."""
        return self.base_fonts.get_scaled_sizes(self.font_scale)
    
    # Legend configurations for each plot
    centers_legend: LegendConfig = field(default_factory=lambda: LegendConfig(
        location='upper right'
    ))
    signal_legend: LegendConfig = field(default_factory=lambda: LegendConfig(
        location='upper right',
        ncol=2,  # Two columns for signal comparison
        framealpha=0.9
    ))
    save_config: SaveConfig = field(default_factory=SaveConfig)