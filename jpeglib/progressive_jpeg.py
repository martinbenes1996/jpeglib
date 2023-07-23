"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import ctypes
from dataclasses import dataclass
import numpy as np
import os
import tempfile
from typing import List, Union
import warnings

from ._bind import CJpegLib
from .spatial_jpeg import SpatialJPEG
from ._cenum import Colorspace, DCTMethod, Dithermode
from ._huffman import Huffman
from . import _infere

@dataclass
class ProgressiveJPEG(SpatialJPEG):
    """JPEG instance to work with progressive JPEG in spatial domain."""
    spatial: List[np.ndarray]
    """pixel data tensor"""
    qt: List[np.ndarray]
    """list of quantization tables per scan"""
    quant_tbl_no: List[List[int]]
    """list of assignments of quantization tables to components per scan"""
    # flags: List[str]
    # """"""
    scan_script: np.ndarray
    """"""

    def _alloc_spatial(self, channels: int = None):
        if channels is None:
            channels = self.color_space.channels
        return (
            (((ctypes.c_ubyte * self.width) * self.height) * channels)
            * self.num_scans
        )()

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
        _spatial = self._alloc_spatial(self.color_space.channels)
        _scan_script = ((ctypes.c_int*9)*self.num_scans)()
        _huffman_bits = ((((ctypes.c_int16*17)*4)*2)*self.num_scans)()
        _huffman_values = ((((ctypes.c_int16*256)*4)*2)*self.num_scans)()
        _qt = (((ctypes.c_ushort * 64) * 4) * self.num_scans)()
        _quant_tbl_no = ((ctypes.c_short * 4) * self.num_scans)()

        # write content into temporary file
        tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        tmp.write(self.content)
        tmp.flush()
        tmp.close()

        # call
        CJpegLib.read_jpeg_progressive(
            path=str(self.path),
            srcfile=tmp.name,
            spatial=_spatial,
            colormap=int(self.jpeg_color_space),
            in_colormap=None,  # support of color quantization
            out_color_space=int(self.color_space),
            dither_mode=dither_mode,
            dct_method=dct_method,
            scan_script=_scan_script,
            huffman_bits=_huffman_bits,
            huffman_values=_huffman_values,
            qt=_qt,
            quant_tbl_no=_quant_tbl_no,
            flags=flags,
        )

        # clean up temporary file
        os.remove(tmp.name)
        # process
        spatial = (
            np.ctypeslib.as_array(_spatial)
            .astype(np.ubyte)
            .reshape(
                self.num_scans,
                self.height,
                -1,
                self.color_space.channels,
            )
        )
        self.spatial = list(spatial)

        # scanscript
        self.scan_script = np.ctypeslib.as_array(_scan_script)
        huffman_bits = np.ctypeslib.as_array(_huffman_bits)
        huffman_values = np.ctypeslib.as_array(_huffman_values)
        # qt
        self.quant_tbl_no = [
            _quant_tbl_no[s][:self.scan_script[s, 0]]
            for s in range(self.num_scans)
        ]
        qt = np.ctypeslib.as_array(_qt).reshape(self.num_scans, -1, 8, 8)
        self.qt = [
            qt[s, :len(np.unique(self.quant_tbl_no[s]))]
            for s in range(self.num_scans)
        ]
        # Huffman
        self.huffmans = []
        for s in range(self.num_scans):
            scan_huffmans = []
            for i in range(4):
                huffman = {
                    k: Huffman(
                        bits=huffman_bits[s, j, i],
                        values=huffman_values[s, j, i]
                    )
                    for k, j in zip(['AC', 'DC'], [1, 0])
                    if huffman_bits[s, j, i, 0] != -1
                }
                scan_huffmans.append(huffman)
            self.huffmans.append(scan_huffmans)

        return self.spatial

    def write_spatial(
        self,
        path: str = None,
        qt: Union[int, np.ndarray] = None,
        quant_tbl_no: np.ndarray = None,
        base_quant_tbl_idx: int = None,
        dct_method: DCTMethod = None,
        # dither_mode: Dithermode = None,
        smoothing_factor: int = None,
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
        # spatial
        spatial = self.spatial[-1]  # taking only the last one
        warnings.warn('writing pixels from the last scan only')
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
                # infere quant_tbl_no
                if quant_tbl_no is None:
                    quant_tbl_no = _infere.quant_tbl_no(qt, spatial=spatial)
                #
                quality, qt = -1, np.ctypeslib.as_ctypes(qt.astype(np.uint16))
                quant_tbl_no = np.ctypeslib.as_ctypes(
                    np.array(quant_tbl_no).astype(np.int16))

        # process
        spatial = np.ctypeslib.as_ctypes(
            spatial.reshape(
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
            base_quant_tbl_idx=base_quant_tbl_idx,
            smoothing_factor=smoothing_factor,
            num_markers=self.num_markers,
            marker_types=self.c_marker_types(),
            marker_lengths=self.c_marker_lengths(),
            markers=self.c_markers(),
            num_scans=self.num_scans,
            scan_script=self.c_scan_script(),
            huffman_bits=self.c_huffman_bits(),
            huffman_values=self.c_huffman_values(),
            flags=flags,
        )


    @property
    def qt(self) -> List[np.ndarray]:
        if self._qt is None:
            self.load()
        return self._qt

    @qt.setter
    def qt(self, qt: List[np.ndarray]):
        self._qt = qt

    @property
    def quant_tbl_no(self) -> List[np.ndarray]:
        if self._quant_tbl_no is None:
            self.load()
        return self._quant_tbl_no

    @quant_tbl_no.setter
    def quant_tbl_no(self, quant_tbl_no: List[np.ndarray]):
        self._quant_tbl_no = quant_tbl_no

    @property
    def scan_script(self) -> np.ndarray:
        return self._scan_script

    @scan_script.setter
    def scan_script(self, scan_script: np.ndarray):
        self._scan_script = scan_script

    def c_scan_script(self):
        return np.ctypeslib.as_ctypes(self.scan_script)

    def c_huffman_bits(self):
        return None

    def c_huffman_values(self):
        return None
