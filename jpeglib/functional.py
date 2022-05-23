
from cmath import inf
from distutils.log import info
import logging
import numpy as np
#from pathlib import Path
import tempfile
import typing

from .dct_jpeg import DCTJPEG
from .spatial_jpeg import SpatialJPEG
from . import _jpeg
from ._colorspace import Colorspace
from ._dctmethod import DCTMethod
from ._dithermode import Dithermode

def read_dct(
    path: str
) -> DCTJPEG:
    """Function for reading of the JPEG DCT coefficients.
    
    The file content is loaded once at the call. Then the operations are independent on the source file.
    
    Reading of DCT is a lossless operation, does not perform the JPEG file decompression.
    
    :param path: Path to a source file in JPEG format.
    :type path: str
    :return: DCT JPEG object
    :rtype: :class:`DCTJPEG`
    :raises [IOError]: When source file does not exist

    :Example:
    
    >>> jpeg = jpeglib.read_dct("input.jpeg")
    >>> jpeg.Y; jpeg.Cb; jpeg.Cr; jpeg.qt
    
    In some cases, libjpeg exits the program. We work on improving this and replace exit with "soft" Python exception.
    
    >>> try:
    >>>     jpeg = jpeglib.read_dct("does_not_exist.jpeg")
    >>> except IOError: # raised, file does not exist
    >>>     pass
    
    After call, only the parameters of JPEG are read, so that it is known, how large buffers need to be allocated.
    The allocation and reading of the data happens on the first query.
    
    >>> jpeg = jpeglib.read_dct("input.jpeg")
    >>> 
    >>> # at this point, internally jpeg.Y is None
    >>> # however on first query, it is read
    >>> print(jpeg.Y) # read and returned
    >>> print(jpeg.Y) # second time it is already stored in the object
    >>> print(jpeg.Cb) # no reading, already stored in the object too
    """
    # load file content
    with open(path, "rb") as f:
        content = f.read()
    # load info
    info = _jpeg.load_jpeg_info(path)
    # create jpeg
    return DCTJPEG(
        path=path,
        content=content,
        height=info.height,
        width=info.width,
        block_dims=info.block_dims,
        samp_factor=info.samp_factor,
        jpeg_color_space=info.jpeg_color_space,
        #num_components=info.num_components,
        quant_tbl_no=None,
        markers=info.markers,
        Y=None,
        Cb=None,
        Cr=None,
        qt=None,
        progressive_mode=info.progressive_mode
    )


def read_spatial(
    path: str,
    out_color_space: str = None,
    dither_mode: str = None,
    dct_method: str = None,
    flags:list = []
) -> SpatialJPEG:
    """Function for decompressing the JPEG as a pixel data (spatial domain).
    
    The file content is loaded once at the call. Then the operations are independent on the source file.
    
    In some cases, libjpeg exits the program. We work on improving this and replace exit with "soft" Python exception.
    
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
    :rtype: :class:`SpatialJPEG`
    :raises [IOError]: When source file does not exist
    
    :Example:
    
    >>> im = jpeglib.read_spatial("input.jpeg")
    >>> im.spatial
    
    >>> try:
    >>>     im = jpeglib.read_spatial("does_not_exist.jpeg")
    >>> except IOError: # raised, file does not exist
    >>>     pass
    
    After call, only the parameters of JPEG are read, so that it is known, how large buffer needs to be allocated.
    The allocation and reading of the data happens on the first query.
    
    >>> im = jpeglib.read_spatial("input.jpeg")
    >>> # at this point, internally jpeg.spatial is None
    >>> # however on first query, it is read
    >>> print(im.spatial) # read and returned
    >>> print(im.spatial) # second time it is already stored in the object 
    """
    # load file content
    with open(path, "rb") as f:
        content = f.read()
    # load info
    info = _jpeg.load_jpeg_info(path)
    # parse
    jpeg_color_space = info.jpeg_color_space
    if out_color_space is not None:
        out_color_space = Colorspace(out_color_space)
        jpeg_color_space = out_color_space
    else:
        out_color_space = Colorspace(
            'JCS_RGB' if info.has_chrominance else 'JCS_GRAYSCALE'
        )

    if dct_method is not None:
        dct_method = DCTMethod(dct_method)
    if dither_mode is not None:
        dither_mode = Dithermode(dither_mode)
    # create jpeg
    return SpatialJPEG(
        path                = path,
        content             = content,
        height              = info.height,
        width               = info.width,
        block_dims          = info.block_dims,
        samp_factor         = info.samp_factor,
        jpeg_color_space    = jpeg_color_space,
        #num_components      = num_components,
        markers             = info.markers,
        spatial             = None,
        color_space         = out_color_space,
        dither_mode         = dither_mode,
        dct_method          = dct_method,
        flags               = flags,
        progressive_mode    = info.progressive_mode
    )

def from_spatial(
    spatial: np.ndarray,
    in_color_space:typing.Union[str,Colorspace] = None
) -> SpatialJPEG:
    """A factory of :class:`SpatialJPEG` from pixel data.
    
    The color space inference is based on number of color channels.
    For single channel, grayscale is assumed. For three channels, rgb is assumed.
    
    .. warning::
        Parameter :obj:`SpatialJPEG.path` is not initialized.
        When calling :meth:`SpatialJPEG.write_spatial`, you have to specify `path`,
        otherwise an error is raised.
        
    :param spatial: Spatial representation.
    :type spatial: np.ndarray
    :param in_color_space: Color space of the input. If not given, infered from the shape.
    :type in_color_space: str | Colorspace, optional
    :raises [IOError]: When color space can't be infered.
    
    :Example:
    
    When data has three color channels (in the dimension 2), rgb is infered.
    
    >>> spatial = np.random.randint(0,255,(16,16,3),dtype=np.uint8)
    >>> im = jpeglib.from_spatial(spatial) # 3 channels -> rgb infered
    >>> print(im.color_space) # -> 'JCS_RGB'
    >>> im.write_spatial("output.jpeg")
    
    When data has one color channels, grayscale is infered
    
    >>> spatial = np.random.randint(0,255,(16,16,1),dtype=np.uint8)
    >>> im = jpeglib.from_spatial(spatial) # 1 channel -> grayscale infered
    >>> print(im.color_space) # -> 'JCS_GRAYSCALE'
    >>> im.write_spatial("output.jpeg")
    
    For other color channels, color space can't be infered. Error is raised.
    
    >>> spatial = np.random.randint(0,255,(16,16,7),dtype=np.uint8)
    >>> try:
    >>>     im = jpeglib.from_spatial(spatial)
    >>> except IOError:
    >>>     raised
    
    When output is not specified when writing, error is raised.
    
    >>> spatial = np.random.randint(0,255,(16,16,3),dtype=np.uint8)
    >>> im = jpeglib.from_spatial(spatial)
    >>> try:
    >>>     im.write_spatial()
    >>> except IOError:
    >>>     pass
    """
    # shape
    height, width, num_components = spatial.shape
    # parse colorspace
    if in_color_space is not None:
        try:
            in_color_space = Colorspace(in_color_space)
        except:
            in_color_space = in_color_space
    # infere colorspace
    if in_color_space is None:
        if num_components == 3:
            in_color_space = Colorspace('JCS_RGB')
            jpeg_color_space = Colorspace('JCS_YCbCr')
        elif num_components == 1:
            in_color_space = Colorspace('JCS_GRAYSCALE')
            jpeg_color_space = Colorspace('JCS_GRAYSCALE')
        else:
            raise IOError('failed to infere colorspace')
    # create jpeg
    return SpatialJPEG(
        path=None,
        content=None,
        height=height,
        width=width,
        block_dims=None,
        samp_factor=None,
        jpeg_color_space=jpeg_color_space,
        markers=None,
        spatial=spatial,
        color_space=in_color_space,
        dither_mode=None,
        dct_method=None,
        flags=[],
        progressive_mode=False
    )

def from_dct(
    Y: np.ndarray,
    Cb: np.ndarray = None,
    Cr: np.ndarray = None,
    qt: np.ndarray = None,
    quant_tbl_no: list = None,
) -> DCTJPEG:
    """A factory of :class:`DCTJPEG` from DCT data.
    
    The color space inference is based on number of color channels.
    If chrominance is not given, grayscale JPEG is assumed. For three channels, YCbCr JPEG is assumed.

    The quantization table assigment is infered based on components and number of quantization tables given.
    If you wish a custom assignment, use :obj:`DCTJPEG.qt`.

    .. warning::
        Parameter :obj:`DCTJPEG.path` is not initialized.
        When calling :meth:`DCTJPEG.write_dct`, you have to specify `path`,
        otherwise an error is raised.
        
    :param Y: Luminance tensor.
    :type Y: np.ndarray
    :param Cb: Blue chrominance tensor.
    :type Cb: np.ndarray
    :param Cr: Red chrominance tensor.
    :type Cr: np.ndarray
    :param qt: Quantization table tensor.
    :type qt: np.ndarray
    
    :Example:
    
    >>> Y = np.random.randint(-127, 127,(2,2,8,8),dtype=np.int16)
    >>> Cb = np.random.randint(-127, 127,(1,1,8,8),dtype=np.int16)
    >>> Cr = np.random.randint(-127, 127,(1,1,8,8),dtype=np.int16)
    >>> im = jpeglib.from_dct(Y, Cb, Cr, qt=75) # chrominance -> YCbCr infered

    When data has one color channels, grayscale is infered
    
    >>> Y = np.random.randint(-127, 127,(2,2,8,8),dtype=np.int16)
    >>> im = jpeglib.from_dct(Y) # chrominance -> YCbCr infered
    
    For other color channels, color space can't be infered. Error is raised.
    
    >>> Y = np.random.randint(-127, 127,(2,2,8,8),dtype=np.int16)
    >>> X = np.random.randint(-127, 127,(1,1,8,8),dtype=np.int16)
    >>> try:
    >>>     im = jpeglib.from_dct(Y, X)
    >>> except IOError:
    >>>     raised
    
    When output is not specified when writing, error is raised.
    
    >>> Y = np.random.randint(-127, 127,(2,2,8,8),dtype=np.int16)
    >>> im = jpeglib.from_dct(Y)
    >>> try:
    >>>     im.write_dct()
    >>> except IOError:
    >>>     pass
    """
    # infere colorspace
    if Cb is not None and Cr is not None:
        jpeg_color_space = Colorspace('JCS_YCbCr')
    elif Cb is None and Cr is None:
        jpeg_color_space = Colorspace('JCS_YCbCr') # only Y
        #jpeg_color_space = Colorspace('JCS_GRAYSCALE')
    else:
        raise IOError('failed to infere colorspace')
    # infere quant_tbl_no
    if quant_tbl_no is None and qt is not None:
        quant_tbl_no = np.array([0,0,0,0])
        if Cb is not None and Cr is not None:
            if qt.shape[0] == 2:
                quant_tbl_no = np.array([0,1,1,0])
            elif qt.shape[0] == 3:
                quant_tbl_no = np.array([0,1,2,0])
    # infere samp_factor
    # TODO
    samp_factor = ((1,1),(1,1),(1,1))
    # block dims
    block_dims = [[Y.shape[0],Y.shape[1]]]
    if Cb is not None and Cr is not None:
        block_dims.append([Cb.shape[0],Cb.shape[1]])
        block_dims.append([Cr.shape[0],Cr.shape[1]])
    block_dims = np.array(block_dims)

    # create jpeg
    return DCTJPEG(
        path=None,
        content=None,
        height=Y.shape[0]*Y.shape[2],
        width=Y.shape[1]*Y.shape[3],
        block_dims=block_dims,
        samp_factor=samp_factor,
        jpeg_color_space=jpeg_color_space,
        #num_components=info.num_components,
        quant_tbl_no=quant_tbl_no,
        markers=None,
        Y=Y,
        Cb=Cb,
        Cr=Cr,
        qt=qt,
        progressive_mode=None,
    )



    # shape
    height, width, num_components = spatial.shape
    # parse colorspace
    if in_color_space is not None:
        try:
            in_color_space = Colorspace(in_color_space)
        except:
            in_color_space = in_color_space
    # infere colorspace
    if in_color_space is None:
        if num_components == 3:
            in_color_space = Colorspace('JCS_RGB')
        elif num_components == 1:
            in_color_space = Colorspace('JCS_GRAYSCALE')
        else:
            raise IOError('failed to infere colorspace')
    # create jpeg
    return SpatialJPEG(
        path=None,
        content=None,
        height=height,
        width=width,
        block_dims=None,
        samp_factor=None,
        jpeg_color_space=None,
        markers=None,
        spatial=spatial,
        color_space=in_color_space,
        dither_mode=None,
        dct_method=None,
        flags=[],
        progressive_mode=False
    )

    # with JPEG(srcfile) as im:
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
