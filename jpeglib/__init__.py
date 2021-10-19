"""
A low-level JPEG loader for acquiring DCT coefficients and quantization tables.
It uses *libjpeg* for reading of the JPEG format.
"""

from ._handle import *
from ._timer import Timer

def set_libjpeg_version(version):
    from ._bind import CJpegLib
    if version in {'6','6b'}:
        CJpegLib._bind_lib(version='6b')
    elif version in {'8','8d'}:
        CJpegLib._bind_lib(version='8d')
    else:
        raise Exception(f'Unsupported libjpeg version! Currently supported versions: 6b 8d')
