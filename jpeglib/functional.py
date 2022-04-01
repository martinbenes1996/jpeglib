
from pathlib import Path
import tempfile
from .jpeg import DCTJPEG,SpatialJPEG
from . import _jpeg

def jpeg_dct(path: str):
    """Reads the DCT JPEG.
    
    :param path: Path to a source file in JPEG format.
    :type path: str
    :return: DCT JPEG object
    :rtype: :class:`dct_jpeg.DCTJPEG`
    :raises [IOError]: When source file does not exist
    
    :Example:
    
    >>> dct = jpeglib.read_dct("input.jpeg")
    """
    # load file content
    with open(path, "rb") as f:
        content = f.read()
    # load info
    info = _jpeg.load_jpeg_info(path)
    # create jpeg
    return DCTJPEG(
        path                = path,
        content             = content,
        height              = info.height,
        width               = info.width,
        block_dims          = info.block_dims,
        samp_factor         = info.samp_factor,
        num_components      = info.num_components,
        jpeg_color_space    = info.jpeg_color_space,
        Y                   = None,
        Cb                  = None,
        Cr                  = None,
        qt                  = None
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
    

    