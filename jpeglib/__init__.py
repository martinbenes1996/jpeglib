"""
Python envelope for the popular C library libjpeg for handling JPEG files.

It offers full control over compression and decompression and exposes DCT coefficients and quantization tables.
"""

import pkg_resources
from .jpeg import *
from .version import *

# for unit tests
from ._timer import Timer

# version
try:
    __version__ = pkg_resources.get_distribution("jpeglib").version
except:
    __version__ = None