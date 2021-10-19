"""
A low-level JPEG loader for acquiring DCT coefficients and quantization tables.
It uses *libjpeg* for reading of the JPEG format.
"""

from . import jpeg
from .jpeg import JPEG
from ._timer import Timer

def set_libjpeg_version(version):
    """Sets the version of libjpeg to use.
    
    Args:
        version (str): Version to use, one of 6b, 8d.
    """
    from ._bind import CJpegLib
    if version in {'6','6b'}:
        CJpegLib._bind_lib(version='6b')
    elif version in {'8','8d'}:
        CJpegLib._bind_lib(version='8d')
    else:
        raise Exception(f'Unsupported libjpeg version! Currently supported versions: 6b 8d')
