"""
Python envelope for the popular C library libjpeg for handling JPEG files.

It offers full control over compression and decompression and exposes DCT coefficients and quantization tables.
"""

# functions
from .functional import *

# jpeg objects
from .dct_jpeg import *
from .spatial_jpeg import *

# cstructs
from ._colorspace import *
from ._dctmethod import *
from ._dithermode import *
from ._marker import *

# libjpeg versions
from .version import *

# DCT implementation
from . import dct

# for unit tests
from ._timer import Timer

# package version
import pkg_resources
try:
    __version__ = pkg_resources.get_distribution("jpeglib").version
except:
    __version__ = None

# set default version
version.set('6b')