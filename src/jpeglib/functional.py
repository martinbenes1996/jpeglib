"""Functional interface of the library.

Global functions to call.

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import numpy as np
import typing
import warnings

from .dct_jpeg import DCTJPEG
from .spatial_jpeg import SpatialJPEG
from .progressive_jpeg import ProgressiveJPEG
from . import _jpeg
from ._cenum import Colorspace, DCTMethod, Dithermode
from ._scan import Scan
from . import _infere


def read_dct(
    path: str
) -> DCTJPEG:
    """Function for reading of the JPEG DCT coefficients.

    The file content is loaded once at the call.
    Then the operations are independent on the source file.

    Reading of DCT is a lossless operation,
    does not perform the JPEG file decompression.

    :param path: Path to a source file in JPEG format.
    :type path: str
    :return: DCT JPEG object
    :rtype: :class:`DCTJPEG`
    :raises [IOError]: When source file does not exist

    :Example:

    >>> jpeg = jpeglib.read_dct("input.jpeg")
    >>> jpeg.Y; jpeg.Cb; jpeg.Cr; jpeg.qt

    In some cases, libjpeg exits the program.
    We work on improving this and replace exit with "soft" Python exception.

    >>> try:
    >>>     jpeg = jpeglib.read_dct("does_not_exist.jpeg")
    >>> except IOError: # raised, file does not exist
    >>>     pass

    After call, only the parameters of JPEG are read, so that it is known,
    how large buffers need to be allocated.
    The allocation and reading of the data happens on the first query.

    >>> jpeg = jpeglib.read_dct("input.jpeg")
    >>>
    >>> # at this point, internally jpeg.Y is None
    >>> # however on first query, it is read
    >>> print(jpeg.Y) # read and returned
    >>> print(jpeg.Y) # second time it is already stored in the object
    >>> print(jpeg.Cb) # no reading, already stored in the object too
    """  # noqa: F501
    # load file content
    path = str(path)
    with open(path, "rb") as f:
        content = f.read()
        f.flush()
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
        num_scans=info.num_scans,
        # num_components=info.num_components,
        quant_tbl_no=None,
        markers=info.markers,
        huffmans=info.huffmans,
        Y=None,
        Cb=None,
        Cr=None,
        K=None,
        qt=None,
        progressive_mode=info.progressive_mode
    )


def read_spatial(
    path: str,
    out_color_space: Colorspace = None,
    dct_method: DCTMethod = None,
    dither_mode: Dithermode = None,
    buffered: bool = False,
    flags: list = None,
) -> typing.Union[SpatialJPEG, ProgressiveJPEG]:
    """Function for decompressing the JPEG as a pixel data (spatial domain).

    The file content is loaded once at the call.
    Then the operations are independent on the source file.

    In some cases, libjpeg exits the program.
    We work on improving this and replace exit with "soft" Python exception.

    :param path: Path to a source file in JPEG format.
    :type path: str
    :param out_color_space: Output color space. If not given, using the libjpeg default.
    :type out_color_space: :class:`Colorspace`
    :param dct_method: DCT method. If not given, using the libjpeg default.
    :type dct_method: :class:`DCTMethod`
    :param dither_mode: Dither mode. If not given, using the libjpeg default.
    :type dither_mode: :class:`Dithermode`
    :param buffered: Read scan by scan. This will return :class:`ProgressiveJPEG` object.
    :type buffered: bool
    :param flags: Bool decompression parameters as str. If not given, using the libjpeg default. Read more at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#flags>`_.
    :type flags: list, optional
    :return: JPEG object
    :rtype: :class:`SpatialJPEG` or :class:`ProgressiveJPEG`
    :raises [IOError]: When source file does not exist

    :Example:

    >>> im = jpeglib.read_spatial("input.jpeg")
    >>> im.spatial

    >>> try:
    >>>     im = jpeglib.read_spatial("does_not_exist.jpeg")
    >>> except IOError: # raised, file does not exist
    >>>     pass

    After call, only the parameters of JPEG are read,
    so that it is known, how large buffer needs to be allocated.
    The allocation and reading of the data happens on the first query.

    >>> im = jpeglib.read_spatial("input.jpeg")
    >>> # at this point, internally jpeg.spatial is None
    >>> # however on first query, it is read
    >>> print(im.spatial) # read and returned
    >>> print(im.spatial) # second time it is already stored in the object

    Read progressive JPEG with

    >>> im = jpeglib.read_spatial("input.jpeg", buffered=True)
    """  # noqa: E501
    # load file content
    path = str(path)
    with open(path, "rb") as f:
        content = f.read()
    # load info
    info = _jpeg.load_jpeg_info(path)
    # parse
    jpeg_color_space = info.jpeg_color_space
    if out_color_space is not None:
        jpeg_color_space = out_color_space
    else:
        if info.has_black:
            out_color_space = 'JCS_CMYK'
        elif info.has_chrominance:
            out_color_space = 'JCS_RGB'
        else:
            out_color_space = 'JCS_GRAYSCALE'
        out_color_space = Colorspace[out_color_space]

    # progressive
    kw = {}
    if buffered:
        jpeg_cls = ProgressiveJPEG
        kw['scans'] = None
    else:
        jpeg_cls = SpatialJPEG
        # if info.progressive_mode:
        #     warnings.warn('loading progressive JPEG as sequential')

    # create jpeg
    im = jpeg_cls(
        path=path,
        content=content,
        height=info.height,
        width=info.width,
        block_dims=info.block_dims,
        samp_factor=info.samp_factor,
        jpeg_color_space=jpeg_color_space,
        num_scans=info.num_scans,
        # num_components = num_components,
        markers=info.markers,
        huffmans=info.huffmans,
        spatial=None,
        color_space=out_color_space,
        progressive_mode=info.progressive_mode,
        **kw,
    )
    # load image data
    if dct_method is not None or dither_mode is not None or flags is not None:
        im.load(
            dct_method=dct_method,
            dither_mode=dither_mode,
            flags=flags,
        )
    return im


def from_spatial(
    spatial: np.ndarray,
    in_color_space: Colorspace = None,
    scans: typing.List[Scan] = None,
) -> typing.Union[SpatialJPEG, ProgressiveJPEG]:
    """A factory of :class:`SpatialJPEG` from pixel data.

    The color space inference is based on number of color channels.
    For single channel, grayscale is assumed.
    For three channels, rgb is assumed.

    .. warning::
        Parameter :obj:`SpatialJPEG.path` is not initialized.
        When calling :meth:`SpatialJPEG.write_spatial`,
        you have to specify `path`,
        otherwise an error is raised.

    :param spatial: Spatial representation.
    :type spatial: np.ndarray
    :param in_color_space: Color space of the input. If not given, infered from the shape.
    :type in_color_space: str | Colorspace, optional
    :raises [IOError]: When color space can't be infered.
    :raises [TypeError]: When spatial has wrong dtype.

    :Example:

    When data has three color channels (in the dimension 2), rgb is infered.

    >>> spatial = np.random.randint(0,255,(16,16,3),dtype=np.uint8)
    >>> im = jpeglib.from_spatial(spatial) # 3 channels -> rgb infered
    >>> print(im.color_space) # -> Colorspace.JCS_RGB
    >>> im.write_spatial("output.jpeg")

    When data has one color channels, grayscale is infered

    >>> spatial = np.random.randint(0,255,(16,16,1),dtype=np.uint8)
    >>> im = jpeglib.from_spatial(spatial) # 1 channel -> grayscale infered
    >>> print(im.color_space) # -> Colorspace.JCS_GRAYSCALE
    >>> im.write_spatial("output.jpeg")

    For other color channels, color space can't be infered. Error is raised.

    >>> spatial = np.random.randint(0,255,(16,16,7),dtype=np.uint8)
    >>> try:
    >>>     im = jpeglib.from_spatial(spatial)
    >>> except IOError:
    >>>     pass

    When spatial has a wrong dtype, TypeError is raised.

    >>> spatial = np.random.randint(0,255,(16,16,7),dtype=np.uint32)
    >>> try:
    >>>     im = jpeglib.from_spatial(spatial)
    >>> except TypeError:
    >>>     pass

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
    if spatial.dtype != np.uint8:
        raise TypeError('spatial must be of type uint8')
    # infere colorspace
    if in_color_space is None:
        in_color_space = _infere.in_color_space(num_components)
    jpeg_color_space = _infere.jpeg_in_color_space(in_color_space)

    #
    kw = {
        'progressive_mode': None,
        'num_scans': 1,
    }
    if scans is None:
        jpeg_cls = SpatialJPEG
    else:
        jpeg_cls = ProgressiveJPEG
        kw['scans'] = scans
        kw['progressive_mode'] = True
        kw['num_scans'] = len(scans)
        spatial = np.expand_dims(spatial, 0)

    # create jpeg
    return jpeg_cls(
        path=None,
        content=None,
        height=height,
        width=width,
        block_dims=None,
        samp_factor=None,
        jpeg_color_space=jpeg_color_space,
        markers=None,
        huffmans=None,
        spatial=spatial,
        color_space=in_color_space,
        **kw,
    )


def from_dct(
    Y: np.ndarray,
    Cb: np.ndarray = None,
    Cr: np.ndarray = None,
    K: np.ndarray = None,
    qt: np.ndarray = None,
    quant_tbl_no: list = None,
) -> DCTJPEG:
    """A factory of :class:`DCTJPEG` from DCT data.

    The color space inference is based on number of color channels.
    If chrominance is not given, grayscale JPEG is assumed.
    For three channels, YCbCr JPEG is assumed.

    The quantization table assigment is infered based on components
    and number of quantization tables given.
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
    :param K: Black tensor.
    :type K: np.ndarray
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
    jpeg_color_space = _infere.jpeg_color_space(Cb, Cr, K)
    # infere quant_tbl_no
    if quant_tbl_no is None:
        quant_tbl_no = _infere.quant_tbl_no(qt, Cb=Cb, Cr=Cr, K=K)
    # infere samp_factor
    samp_factor = _infere.samp_factor(Y, Cb, Cr, K)
    # block dims
    block_dims = _infere.block_dims(Y, Cb, Cr, K)

    # create jpeg
    return DCTJPEG(
        path=None,
        content=None,
        height=Y.shape[0]*Y.shape[2],
        width=Y.shape[1]*Y.shape[3],
        block_dims=block_dims,
        samp_factor=samp_factor,
        jpeg_color_space=jpeg_color_space,
        num_scans=1,
        # num_components=info.num_components,
        quant_tbl_no=quant_tbl_no,
        markers=None,
        huffmans=None,
        Y=Y,
        Cb=Cb,
        Cr=Cr,
        K=None,
        qt=qt,
        progressive_mode=None,
    )
