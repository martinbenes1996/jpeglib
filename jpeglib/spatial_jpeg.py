
from dataclasses import dataclass
import numpy as np
from ._jpeg import JPEG

@dataclass
class SpatialJPEG(JPEG):
    """JPEG instance to work in spatial domain."""
    x: np.ndarray
    color_space: str
