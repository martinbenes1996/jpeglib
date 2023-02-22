"""
Python envelope for the popular C library libjpeg for handling JPEG files.

It offers full control over compression and decompression
and exposes DCT coefficients and quantization tables.
"""

# functions
from .functional import read_dct, read_spatial, from_spatial, from_dct

# jpeg objects
from .dct_jpeg import DCTJPEG, DCTJPEGio, to_jpegio
from .spatial_jpeg import SpatialJPEG

# cstructs
from ._colorspace import Colorspace
from ._dctmethod import DCTMethod
from ._dithermode import Dithermode
from ._marker import Marker

# libjpeg versions
from .version import version

# for unit tests
from ._notations import Jab_to_factors

# package version
import pkg_resources
try:
    __version__ = pkg_resources.get_distribution("jpeglib").version
except pkg_resources.DistributionNotFound:
    __version__ = None

# set default version
try:
    version.set('6b')
except IndexError:
    import logging
    logging.warning('found versions: ' + ' '.join(version.versions()))
    raise RuntimeError('invalid installation, version 6b not found')

__all__ = [
    'read_dct', 'read_spatial', 'from_spatial', 'from_dct', 'to_jpegio',
    'SpatialJPEG', 'DCTJPEG', 'DCTJPEGio',
    'Colorspace', 'DCTMethod', 'Dithermode', 'Marker',
    'version', 'Jab_to_factors',
    '__version__',
]
