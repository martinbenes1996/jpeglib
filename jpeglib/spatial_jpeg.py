
import ctypes
from dataclasses import dataclass
import numpy as np
import os
import tempfile
import typing
import warnings

from ._bind import CJpegLib
from ._jpeg import JPEG
from ._colorspace import Colorspace
from ._dithermode import Dithermode
from ._dctmethod import DCTMethod


@dataclass
class SpatialJPEG(JPEG):
    """JPEG instance to work in spatial domain."""
    spatial: np.ndarray
    """pixel data tensor"""
    color_space: Colorspace
    """color space of the pixel data"""
    # needed for compression, not actual props
    dither_mode: Dithermode
    dct_method: DCTMethod
    flags: list

    def _alloc_spatial(self, channels: int = None):
        if channels is None:
            channels = self.color_space.channels
        return (((ctypes.c_ubyte * self.width) * self.height) * channels)()

    def load(self) -> np.ndarray:

        # colorspace
        if self.color_space is None:
            self.color_space = Colorspace('JCS_RGB')
            # self.color_space = self.jpeg_color_space
        # dither mode
        dither_mode = None
        if self.dither_mode is not None:
            dither_mode = self.dither_mode.index
        # dct method
        dct_method = None
        if self.dct_method is not None:
            dct_method = self.dct_method.index

        # allocate spatial
        spatial = self._alloc_spatial(self.color_space.channels)

        # write content into temporary file
        tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        tmp.write(self.content)
        tmp.flush()
        tmp.close()

        # call
        CJpegLib.read_jpeg_spatial(
            path=str(self.path),
            srcfile=tmp.name,
            spatial=spatial,
            colormap=self.jpeg_color_space.index,
            in_colormap=None,  # support of color quantization
            out_color_space=self.color_space.index,
            dither_mode=dither_mode,
            dct_method=dct_method,
            flags=self.flags,
        )
        # clean up temporary file
        os.remove(tmp.name)
        # process
        self.spatial = (
            np.ctypeslib.as_array(spatial)
            .astype(np.ubyte)
            .reshape(self.height, -1, self.color_space.channels)
        )

        return self.spatial

    def read_spatial(self) -> np.ndarray:
        warnings.warn('read_spatial() is obsolete, use load()')
        return self.load()

    def write_spatial(self,
                      path: str = None,
                      qt: typing.Union[int, np.ndarray] = None,
                      quant_tbl_no: np.ndarray = None,
                      dct_method: typing.Union[str, DCTMethod] = None,
                      # dither_mode: Dithermode = None,
                      smoothing_factor: int = None,
                      flags: list = []):
        """Writes a spatial image representation (i.e. RGB) to a file.

        :param path: Destination file name. If not given, source file is overwritten.
        :type path: str, optional
        :param qt: Compression quality, can be integer 0-100 or a tensor with quantization tables. Defaultly -1 (default factor kept).
        :type qt: int | numpy.ndarray, optional
        :param quant_tbl_no: assignment of quantization tables to components, (0 Y, 1 Cb, 1Cr) by default
        :type quant_tbl_no: numpy.ndarray, optional
        :param dct_method: DCT method, must be accepted by :class:`_dctmethod.DCTMethod`. If not given, using the libjpeg default.
        :type dct_method: str | :class:`_dctmethod.DCTMethod`, optional
        :param smoothing_factor: Smoothing factor, between 0 and 100. Using default from libjpeg by default.
        :type smoothing_factor: int, optional
        :param flags: Bool compression parameters as str. If not given, using the libjpeg default. More at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#flags>`_.
        :type flags: list, optional

        :Example:

        >>> jpeg = jpeglib.read_spatial("input.jpeg")
        >>> jpeg.write_spatial("output.jpeg", qt=75)

        Grayscale JPEG (only lumo channel) using libjpeg color conversion

        >>> x = np.random.randint(0,255,(16,16,3),dtype=np.uint8)
        >>> im = jpeglib.from_spatial(x)
        >>> im.jpeg_color_space = jpeglib.Colorspace('JCS_GRAYSCALE')
        >>> im.write_spatial("hello.jpeg")
        """  # noqa: E501
        # colorspace
        if self.jpeg_color_space is not None:
            jpeg_color_space = self.jpeg_color_space
        else:
            jpeg_color_space = self.color_space
        num_components = (ctypes.c_int*2)(
            self.color_space.channels, jpeg_color_space.channels)
        jpeg_color_space = (ctypes.c_int*2)(
            self.color_space.index, jpeg_color_space.index)
        # path
        dstfile = path if path is not None else self.path
        if dstfile is None:
            raise IOError('no destination file specified')
        # dct method
        if dct_method is not None:
            # str
            try:
                dct_method = DCTMethod(dct_method).index
            # DCTMethod
            except KeyError:
                dct_method = dct_method.index
        # quality
        # use default of library
        if qt is None:
            quality, qt = -1, None
        else:
            # quality factor
            try:
                quality, qt = int(qt), None
            # quantization table
            except TypeError:
                quality, qt = -1, np.ctypeslib.as_ctypes(qt.astype(np.uint16))
                if quant_tbl_no is not None:
                    quant_tbl_no = np.ctypeslib.as_ctypes(np.array(quant_tbl_no).astype(np.int16))

        # process
        spatial = np.ctypeslib.as_ctypes(
            self.spatial.reshape(
                self.color_space.channels,
                self.height,
                self.width,
            )
        )
        # call
        CJpegLib.write_jpeg_spatial(
            dstfile=str(dstfile),
            spatial=spatial,
            image_dims=self.c_image_dims(),
            jpeg_color_space=jpeg_color_space,
            num_components=num_components,
            dct_method=dct_method,
            samp_factor=self.c_samp_factor(),
            qt=qt,
            quality=quality,
            quant_tbl_no=quant_tbl_no,
            smoothing_factor=smoothing_factor,
            num_markers=self.num_markers,
            marker_types=self.c_marker_types(),
            marker_lengths=self.c_marker_lengths(),
            markers=self.c_markers(),
            flags=flags,
        )

    @property
    def spatial(self) -> np.ndarray:
        if self._spatial is None:
            self.load()
        return self._spatial

    @spatial.setter
    def spatial(self, spatial: np.ndarray):
        self._spatial = spatial

    @property
    def color_space(self) -> np.ndarray:
        return self._color_space

    @color_space.setter
    def color_space(self, color_space: Colorspace):
        self._color_space = color_space

    @property
    def channels(self) -> int:
        if self._color_space is None:
            return None
        return self._color_space.channels

    @property
    def dither_mode(self) -> np.ndarray:
        return self._dither_mode

    @dither_mode.setter
    def dither_mode(self, dither_mode: Dithermode):
        self._dither_mode = dither_mode

    @property
    def dct_method(self) -> np.ndarray:
        return self._dct_method

    @dct_method.setter
    def dct_method(self, dct_method: DCTMethod):
        self._dct_method = dct_method

    @property
    def flags(self) -> list:
        return self._flags

    @flags.setter
    def flags(self, flags: list):
        self._flags = flags

    def free(self):
        """Free the allocated tensors."""
        del self._spatial
