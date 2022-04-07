
import logging
import numpy as np
#from pathlib import Path
import tempfile
import typing
from .dct_jpeg import DCTJPEG
from .spatial_jpeg import SpatialJPEG
from . import _jpeg
from ._colorspace import Colorspace

def read_dct(path: str):
    """Reads the DCT JPEG.
    
    :param path: Path to a source file in JPEG format.
    :type path: str
    :return: DCT JPEG object
    :rtype: :class:`dct_jpeg.DCTJPEG`
    :raises [IOError]: When source file does not exist
    
    :Example:
    
    >>> jpeg = jpeglib.read_jpeg_dct("input.jpeg")
    """
    # load file content
    with open(path, "rb") as f:
        content = f.read()
    # load info
    logging.error("_jpeg.load_jpeg_info")
    info = _jpeg.load_jpeg_info(path)
    logging.error("constructing DCTJPEG")
    # create jpeg
    return DCTJPEG(
        path                = path,
        content             = content,
        height              = info.height,
        width               = info.width,
        block_dims          = info.block_dims,
        samp_factor         = info.samp_factor,
        jpeg_color_space    = info.jpeg_color_space,
        num_components      = info.num_components,
        markers             = info.markers,
        Y                   = None,
        Cb                  = None,
        Cr                  = None,
        qt                  = None,
    )


    samp_factor: np.ndarray
    jpeg_color_space: Colorspace
    num_components: int

def read_spatial(
    path: str,
    out_color_space: str = None,
    dither_mode: str = None,
    dct_method: str = None,
    flags:list = []):
    """Decompresses the file into the spatial domain. 
    
    :param path: Path to a source file in JPEG format.
    :type path: str
    :param out_color_space: Output color space, must be accepted by :class:`_colorspace.Colorspace`. If not given, using the libjpeg default.
    :type out_color_space: str, optional
    :param dither_mode: Dither mode, must be accepted by :class:`_dithermode.Dithermode`. If not given, using the libjpeg default.
    :type dither_mode: str, optional
    :param dct_method: DCT method, must be accepted by :class:`_dctmethod.DCTMethod`. If not given, using the libjpeg default.
    :type dct_method: str, optional
    :param flags: Bool decompression parameters as str. If not given, using the libjpeg default. Read more at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#flags>`_.
    :type flags: list, optional
    :return: Spatial JPEG object
    :rtype: :class:`spatial_jpeg.SpatialJPEG`
    :raises [IOError]: When source file does not exist

    :Example:
    
    >>> jpeg = jpeglib.read_jpeg_spatial("input.jpeg")
    """
    # load file content
    with open(path, "rb") as f:
        content = f.read()
    # load info
    info = _jpeg.load_jpeg_info(path)
    # create jpeg
    return SpatialJPEG(
        path                = path,
        content             = content,
        height              = info.height,
        width               = info.width,
        block_dims          = info.block_dims,
        samp_factor         = info.samp_factor,
        jpeg_color_space    = info.jpeg_color_space,
        num_components      = info.num_components,
        markers             = info.markers,
        spatial             = None,
        color_space         = out_color_space,
        dither_mode         = dither_mode,
        dct_method          = dct_method,
        flags               = flags,
    )

def from_spatial(spatial: np.ndarray, in_color_space:typing.Union[str,Colorspace] = None):
    """Factory of spatial JPEG from spatial representation.
    
    Does not set the source path, you have to specify dstfile when writing.
        
    :param spatial: Spatial representation.
    :type spatial: np.ndarray
    :param in_color_space: Color space of the input. If not given, infered from the shape.
    :type in_color_space: str | Colorspace, optional
    
    :Example:
    
    >>> spatial = np.random.randint(0,255,(16,16,3),dtype=np.uint8)
    >>> jpeg = jpeglib.from_spatial(spatial)
    """
    # shape
    height,width,num_components = spatial.shape
    # parse colorspace
    if in_color_space is not None:
        try:
            color_space = Colorspace(in_color_space)
        except:
            color_space = in_color_space
    # infere colorspace
    if color_space is None:
        if num_components == 3:
            color_space = Colorspace('JCS_RGB')
        elif num_components == 1:
            color_space = Colorspace('JCS_GRAYSCALE')
        else:
            raise IOError('failed to infere colorspace')
    # create jpeg
    return cls(
        path                = None,
        content             = None,
        height              = height,
        width               = width,
        block_dims          = None,
        samp_factor         = None,
        jpeg_color_space    = None,
        num_components      = num_components,
        markers             = None,
        spatial             = spatial,
        color_space         = color_space,
        dither_mode         = None,
        dct_method          = None,
        flags               = [],
    )
    
    #with JPEG(srcfile) as im:
    #    Y,CbCr,qt = im.read_dct(*args, **kwargs)

# def write_dct(Y, CbCr=None, dstfile=None, qt=None, *args, **kwargs):
#     """Writes DCT coefficients to a file.
    
#     :param srcfile: Path to a source file in JPEG format. An implicit destination for writing.
#     :type srcfile: str
#     :param args: Positional arguments passed to JPEG.write_dct.
#     :param kwargs: Named arguments passed to JPEG.write_dct.
    
#     More information in documentation of JPEG.write_dct.
    
#     :Example:
    
#     >>> Y,CbCr,qt = jpeglib.read_dct("input.jpeg")
#     >>> jpeglib.write_dct(Y, CbCr, "output.jpeg", qt=qt)
#     """
#     assert(dstfile is not None)
#     with JPEG() as im:
#         im.write_dct(Y, CbCr, dstfile=dstfile, qt=qt, *args, **kwargs)

# def read_spatial(srcfile, *args, **kwargs):
#     """Decompresses the file into the spatial domain.
    
#     :param srcfile: Path to a source file in JPEG format. An implicit destination for writing.
#     :type srcfile: str
#     :param args: Positional arguments passed to JPEG.read_spatial.
#     :param kwargs: Named arguments passed to JPEG.read_spatial.
    
#     More information in documentation of JPEG.read_spatial.
    
#     :Example:
    
#     >>> x = jpeglib.read_spatial("input.jpeg")
#     >>> import matplotlib.pyplot as plt; plt.imshow(x)
#     """
#     with JPEG(srcfile) as im:
#         x = im.read_spatial(*args, **kwargs)
#         return x

# def write_spatial(data, dstfile, *args, **kwargs):
#     """Writes spatial image representation (i.e. RGB) to a file.
    
#     :param data: Numpy array with spatial data.
#     :type data: numpy.ndarray
#     :param dstfile: Destination file name. If not given, source file is overwritten.
#     :type dstfile: str, optional
#     :param args: Positional arguments passed to JPEG.write_spatial.
#     :param kwargs: Named arguments passed to JPEG.write_spatial.
    
#     More information in documentation of JPEG.read_spatial.
    
#     :Example:
    
#     >>> x = jpeglib.read_spatial("input.jpeg")
#     >>> jpeglib.write_spatial(x, "output.jpeg")
#     """
#     with JPEG() as im:
#         im.write_spatial(data, dstfile, *args, **kwargs)
    

    