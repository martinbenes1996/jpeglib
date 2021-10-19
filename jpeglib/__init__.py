"""
A low-level JPEG loader for acquiring DCT coefficients and quantization tables.
It uses *libjpeg* for reading of the JPEG format.
"""

from . import _handle
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

class JPEG(_handle.JPEG):
    """Class representing the JPEG object.
    
    Attributes:
        srcfile (str): Source file name.
        channels (int): Number of channels.
        color_space (str): Output color space.
    """
    def __init__(self, srcfile):
        """Object constructor.
        
        Args:
            srcfile (str): File to read metadata from.
        """
        super().__init__(srcfile)
    
    def read_dct(self):
        """Reads the file DCT."""
        return super().read_dct()

    def write_dct(self, dstfile, Y=None, CbCr=None): # qt, 
        """Writes DCT coefficients to the file.
        
        Args:
            dstfile (str): Destination file name.
            Y (np.array, optional): Modified lumo tensor.
            CbCr (np.array, optional): Modified chroma tensor.
        """
        return super().write_dct(
            dstfile = dstfile, 
            Y       = Y,
            CbCr    = CbCr
        )

    def read_spatial(self, out_color_space=None, dither_mode=None, dct_method=None,
                     flags=[]):
        """Reads the file in spatial domain.
        
        Args:
            out_color_space (str, optional):    Output color space. Must be key of J_COLOR_SPACE.
            dither_mode (str,optional):         Dither mode. Must be key of J_DITHER_MODE.
                                                Using default from libjpeg by default.
            dct_method (str, optional):         DCT method. Must be key of J_DCT_METHOD.
                                                Using default from libjpeg by default.
            flags (list, optional):             Bool decompression parameters as str to set to be true.
                                                Using default from libjpeg by default.
        """
        super().read_spatial(
            out_color_space = out_color_space, 
            dither_mode     = dither_mode, 
            dct_method      = dct_method, 
            flags           = flags
        )

    def write_spatial(self, dstfile, data=None, in_color_space=None, dct_method="JDCT_ISLOW",
                      samp_factor=None, quality=100, smoothing_factor=None, flags=[]):
        """Writes spatial image representation (i.e. RGB) to a file.
        
        Args:
            dstfile (str):                      Destination file name.
            data (np.array):                    Numpy array with spatial data.
            in_color_space (str, optional):     Input color space. Must be key of J_COLOR_SPACE.
                                                JCS_RGB if data given, otherwise from source.
            dct_method (str, optional):         DCT method. Must be key of J_DCT_METHOD.
                                                Using default from libjpeg by default.
            samp_factor (tuple, optional):      Sampling factor. None, tuple of three ints or tuples of two ints.
                                                According to source by default.
            quality (int, optional)             Compression quality, between 0 and 100.
                                                Defaultly 100 (full quality).
            smoothing_factor (int, optional):   Smoothing factor, between 0 and 100.
                                                Using default from libjpeg by default.
            flags (list, optional):             Bool decompression parameters as str to set to be true.
                                                Using default from libjpeg by default.
        """
        super().write_spatial(
            dstfile          = dstfile,
            data             = data,
            in_color_space   = in_color_space,
            dct_method       = dct_method,
            samp_factor      = samp_factor,
            quality          = quality,
            smoothing_factor = smoothing_factor,
            flags            = flags
        )

    def to_spatial(self, Y=None, CbCr=None, **kw): #, qt=None):
        """Converts DCT representation to RGB. Uses temporary file to compress and decompress."""
        super().to_spatial(
            Y    = Y,
            CbCr = CbCr,
            **kw
        )
