"""
Python envelope for the popular C library libjpeg for handling JPEG files.

It offers full control over compression and decompression
and exposes DCT coefficients and quantization tables.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

# functions
from .functional import read_dct, read_spatial, from_spatial, from_dct
from .functional import quantize, qf_to_qt
from ._notations import Jab_to_factors

# jpeg objects
from .dct_jpeg import DCTJPEG, DCTJPEGio, to_jpegio
from .spatial_jpeg import SpatialJPEG
from .progressive_jpeg import ProgressiveJPEG

# libjpeg versions
from .version import version

# cenums
from ._cenum import Colorspace, DCTMethod, Dithermode, MarkerType
from ._huffman import Huffman
from ._marker import Marker
from ._scan import Scan
# cenum abbreviations
JCS_UNKNOWN = Colorspace.JCS_UNKNOWN
JCS_GRAYSCALE = Colorspace.JCS_GRAYSCALE
JCS_RGB = Colorspace.JCS_RGB
JCS_YCbCr = Colorspace.JCS_YCbCr
JCS_CMYK = Colorspace.JCS_CMYK
JCS_YCCK = Colorspace.JCS_YCCK
JDCT_ISLOW = DCTMethod.JDCT_ISLOW
JDCT_IFAST = DCTMethod.JDCT_IFAST
JDCT_FLOAT = DCTMethod.JDCT_FLOAT
JPEG_APP0 = MarkerType.JPEG_APP0
JPEG_APP1 = MarkerType.JPEG_APP1
JPEG_COM = MarkerType.JPEG_COM

# package version
import importlib.metadata
try:
    __version__ = importlib.metadata.version("jpeglib")
except importlib.metadata.PackageNotFoundError:
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
    'quantize', 'qf_to_qt',
    'SpatialJPEG', 'DCTJPEG', 'DCTJPEGio', 'ProgressiveJPEG',
    'Colorspace', 'DCTMethod', 'Dithermode', 'Marker', 'MarkerType', 'Huffman', 'Scan',
    'version', 'Jab_to_factors',
    '__version__',
]
