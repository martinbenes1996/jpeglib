"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import ctypes
from dataclasses import dataclass
import numpy as np
import os
import pathlib
import struct
import tempfile
from typing import Union, List
import warnings

from ._bind import CJpegLib
from ._cenum import Colorspace, Dithermode, DCTMethod
from . import _infere
from ._jpeg import JPEG
from .version import version


@dataclass
class SpatialJPEG(JPEG):
    """JPEG instance to work in spatial domain."""
    spatial: np.ndarray
    """pixel data tensor"""
    color_space: Colorspace
    """color space of the pixel data"""
    # flags: List[str]

    def _alloc_spatial(self, channels: int = None):
        if channels is None:
            channels = self.color_space.channels
        return (((ctypes.c_ubyte * self.width) * self.height) * channels)()

    def load(
        self,
        dct_method: DCTMethod = None,
        dither_mode: Dithermode = None,
        flags: List[str] = None,
    ) -> np.ndarray:

        # colorspace
        if self.color_space is None:
            if self.jpeg_color_space == Colorspace('JCS_CMYK'):
                self.color_space = Colorspace('JCS_CMYK')
            else:
                self.color_space = Colorspace('JCS_RGB')
        # dithermode
        if dither_mode:
            dither_mode = int(dither_mode)
        # dct method
        if dct_method:
            dct_method = int(dct_method)

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
            colormap=int(self.jpeg_color_space),
            in_colormap=None,  # support of color quantization
            out_color_space=int(self.color_space),
            dither_mode=dither_mode,
            dct_method=dct_method,
            flags=flags,
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

    def write_spatial(
        self,
        path: str = None,
        qt: Union[int, np.ndarray] = None,
        quant_tbl_no: np.ndarray = None,
        base_quant_tbl_idx: int = None,
        dct_method: DCTMethod = None,
        # dither_mode: Dithermode = None,
        smoothing_factor: int = None,
        dst_unquantized: str = None,
        flags: List[str] = []
    ):
        """Writes a spatial image representation (i.e. RGB) to a file.

        :param path: Destination file name. If not given, source file is overwritten.
        :type path: str, optional
        :param qt: Compression quality, can be integer 0-100 or a tensor with quantization tables. Defaultly -1 (default factor kept).
        :type qt: int | numpy.ndarray, optional
        :param quant_tbl_no: assignment of quantization tables to components, (0 Y, 1 Cb, 1 Cr) by default
        :type quant_tbl_no: numpy.ndarray, optional
        :param base_quant_tbl_idx: base QT to scale, only supported in MozJPEG 3+
        :type base_quant_tbl_idx: int, optional
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
        >>> im.jpeg_color_space = jpeglib.JCS_GRAYSCALE
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
            int(self.color_space), int(jpeg_color_space))
        # dct method
        if dct_method is not None:
            dct_method = int(dct_method)
        # path
        dstfile = path if path is not None else self.path
        if dstfile is None:
            raise IOError('no destination file specified')
        # scans
        if self.num_scans > 1:
            warnings.warn('saving progressive JPEG as sequential, specify buffered=True in read_spatial to stay progressive')
        # quality
        # use default of library
        if qt is None:
            quality, qt = -1, None
        else:
            # quality factor
            try:
                quality, qt = int(qt), None
                if quality < 1 or quality > 100:
                    warnings.warn('quality factor clipped to 1-100')
                quality = np.clip(quality, 1, 100)

            # quantization table
            except TypeError:
                # infere quant_tbl_no
                if quant_tbl_no is None:
                    quant_tbl_no = _infere.quant_tbl_no(qt, spatial=self.spatial)
                #
                quality, qt = -1, np.ctypeslib.as_ctypes(qt.astype(np.uint16))
                quant_tbl_no = np.ctypeslib.as_ctypes(np.array(quant_tbl_no).astype(np.int16))

        # process
        spatial = np.ctypeslib.as_ctypes(self.spatial.flatten())

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
            base_quant_tbl_idx=base_quant_tbl_idx,
            smoothing_factor=smoothing_factor,
            num_markers=self.num_markers,
            marker_types=self.c_marker_types(),
            marker_lengths=self.c_marker_lengths(),
            markers=self.c_markers(),
            num_scans=1,
            scan_script=None,
            huffman_bits=self.c_huffman_bits(),
            huffman_values=self.c_huffman_values(),
            progressive_mode=self.progressive_mode,
            dst_unquantized=dst_unquantized,
            flags=flags,
        )

    def unquantized_coefficients(
        self,
        dct_method: DCTMethod = None,
        flags: List[str] = [],
    ) -> List[np.ndarray]:
        """Get unquantized coefficients.

        :param dct_method: DCT method, must be accepted by :class:`_dctmethod.DCTMethod`. If not given, using the libjpeg default.
        :type dct_method: str | :class:`_dctmethod.DCTMethod`, optional
        :param flags: Compression flags.
        :type flags: list
        :return: Bool compression parameters as str. If not given, using the libjpeg default. More at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#flags>`_.
        :rtype: list

        Quick-and-dirty implementation.
        Current TODOs are JDCT_IFAST method and other samp_factors
        """
        # implementation limits
        assert self.samp_factor == '4:4:4' or (self.samp_factor == 1).all()
        assert version.get() in version.LIBJPEG_VERSIONS
        # write to temporary files
        # tmp_dir = pathlib.Path(tempfile.mkdtemp())
        # dst = str(tmp_dir / 'dst.jpeg')
        # dst_uq = str(tmp_dir / 'uq.bin')
        dst = tempfile.NamedTemporaryFile('w', suffix='.jpeg', delete=False)
        pathlib.Path(dst.name).parent.mkdir(parents=True, exist_ok=True)
        dst_uq = tempfile.NamedTemporaryFile('w', suffix='.bin', delete=False)
        pathlib.Path(dst_uq.name).parent.mkdir(parents=True, exist_ok=True)
        # dst.close()
        # dst_uq.close()
        self.write_spatial(
            path=dst.name,
            dct_method=dct_method,
            dst_unquantized=dst_uq.name,
            flags=flags
        )

        # extract unquantized coefficients
        print('opening', dst_uq.name)
        with open(dst_uq.name, 'rb') as fp:
            content = fp.read()
        os.remove(dst.name)
        os.remove(dst_uq.name)
        uq = struct.unpack(f'<{len(content)//4}f', content)
        uq = np.reshape(uq, (
            self.height//8, self.width//8,
            self.num_components,
            8, 8,
        ))
        return np.moveaxis(uq, 2, 0)

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
        try:
            return self._color_space.channels
        except AttributeError:
            return None

    @property
    def flags(self) -> list:
        return self._flags

    @flags.setter
    def flags(self, flags: list):
        self._flags = flags

    def free(self):
        """Free the allocated tensors."""
        del self._spatial
