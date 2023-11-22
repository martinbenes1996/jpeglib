"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import ctypes
import dataclasses
import numpy as np
import os
import tempfile
from typing import List, Tuple
import warnings

from ._bind import CJpegLib
from . import _jpeg


@dataclasses.dataclass
class DCTJPEG(_jpeg.JPEG):
    """JPEG instance to work with DCT domain."""

    Y: np.ndarray
    """luminance tensor with shape [num_vertical_blocks, num_horizontal_blocks, vertical_block_size, horizontal_block_size]"""  # noqa: E501
    Cb: np.ndarray
    """chrominance blue-difference tensor with shape [num_vertical_blocks, num_horizontal_blocks, vertical_block_size, horizontal_block_size]"""  # noqa: E501
    Cr: np.ndarray
    """chrominance red-difference tensor with shape [num_vertical_blocks, num_horizontal_blocks, vertical_block_size, horizontal_block_size]"""  # noqa: E501
    K: np.ndarray
    """black (cmyK) tensor"""
    qt: np.ndarray
    """quantization tensor with shape [vertical_block_size, horizontal_block_size]"""  # noqa: E501
    quant_tbl_no: np.ndarray
    """assignment of quantization tables to components,
    (0 Y, 1 Cb, 1Cr) by default"""

    def _alloc_dct_component(self, i: int):
        return (
            ((ctypes.c_short * 64) *
             self.width_in_blocks(i)) *
            self.height_in_blocks(i)
        )()

    def load(self) -> Tuple:
        """Function to allocate the buffer and read the image data.

        :return:
        luminance Y and chrominance Cb, Cr tensors
        and quantization table qt as Y,(Cb,Cr),qt
        :rtype: tuple

        :Example:

        >>> jpeg = jpeglib.read_dct("input.jpeg")
        >>> Y,(Cb,Cr),qt = jpeg.load()

        Function is internally called when

        >>> jpeg = jpeglib.read_dct("input.jpeg")
        >>> jpeg.Y # jpeg.load() called internally
        """
        # no content
        if self.content is None:
            return None, (None, None), None

        # allocate DCT components
        Y = self._alloc_dct_component(0)
        Cb, Cr = None, None
        if self.has_chrominance:  # has chrominance
            Cb = self._alloc_dct_component(1)
            Cr = self._alloc_dct_component(2)
        K = None
        if self.has_black:  # has black
            K = self._alloc_dct_component(3)
        qt = ((ctypes.c_ushort * 64) * 4)()
        _quant_tbl_no = (ctypes.c_short * 4)()

        # write content into temporary file
        tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        tmp.write(self.content)
        tmp.flush()
        tmp.close()

        # call
        CJpegLib.read_jpeg_dct(
            path=str(self.path),
            srcfile=tmp.name,
            Y=Y,
            Cb=Cb,
            Cr=Cr,
            K=K,
            qt=qt,
            quant_tbl_no=_quant_tbl_no,
        )
        # clean up temporary file
        os.remove(tmp.name)

        # process
        def process_component(comp):
            comp_np = np.ctypeslib.as_array(comp)
            return comp_np.reshape((*comp_np.shape[:-1], 8, 8))
        qt = process_component(qt)
        if self._Y is None:
            self.Y = process_component(Y)
        if self.has_chrominance:
            if self._Cb is None:
                self.Cb = process_component(Cb)
            if self._Cr is None:
                self.Cr = process_component(Cr)
        if self.has_black:
            if self._K is None:
                self.K = process_component(K)
        if self._quant_tbl_no is None:
            self.quant_tbl_no = np.array([
                _quant_tbl_no[i]
                for i in range(self.num_components)
            ])
        # crop
        if self._qt is None:
            self.qt = qt[:np.max(self.quant_tbl_no)+1]
        # return
        return self.Y, (self.Cb, self.Cr), self.qt

    def read_dct(self) -> Tuple:
        warnings.warn('read_dct() is obsolete, use load()')
        return self.load()

    def write_dct(
        self,
        path: str = None,
        quality: int = -1,
        flags: List[str] = [],
    ):
        """Function to write DCT coefficients to JPEG file.

        Does not perform JPEG compression, writing DCT is lossless.

        :param path: Destination file name. If not given, source file is overwritten.
        :type path: str, optional
        :param quality: Compression quality, between 0 and 100. Special value -1 stands for using qt inside the instance or keeping libjpeg default.
        :type quality: int, optional
        :param flags: Bool compression parameters as str. If not given, using the libjpeg default. More at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#flags>`_.
        :type flags: list, optional

        :Example:

        >>> jpeg = jpeglib.read_spatial("input.jpeg")
        >>> jpeg.write_spatial("output.jpeg", quality=92)
        """  # noqa: E501
        # path
        dstfile = path if path is not None else self.path
        if dstfile is None:
            raise IOError('no destination file specified')

        # convert dct
        def process_component(comp):
            if comp is None:
                return None
            comp = comp.reshape((*comp.shape[:-2], 64))
            return np.ctypeslib.as_ctypes(comp.astype(np.int16))
        if isinstance(self.qt, int):
            qt = None
            if quality != -1:
                quality = self.qt
        else:
            qt = process_component(self.qt)
        quant_tbl_no = np.array([-1]*4)
        if self.quant_tbl_no is not None:
            for i, q in enumerate(self.quant_tbl_no):
                quant_tbl_no[i] = q
            quant_tbl_no = np.ctypeslib.as_ctypes(quant_tbl_no.astype(np.int16))
        else:
            quant_tbl_no = None
        Y = process_component(self.Y)
        Cb = process_component(self.Cb)
        Cr = process_component(self.Cr)
        K = process_component(self.K)
        # quality and quantization table
        assert quality in set(range(-1, 101))
        if quality != -1:
            qt = None

        tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        if self.content is not None:
            # write content into temporary file
            tmp.write(self.content)
            tmp.flush()
        tmp.close()

        # call
        CJpegLib.write_jpeg_dct(
            srcfile=tmp.name if self.content is not None else None,
            dstfile=str(dstfile),
            Y=Y,
            Cb=Cb,
            Cr=Cr,
            K=K,
            image_dims=self.c_image_dims(),
            block_dims=self.c_block_dims(),
            samp_factor=self.c_samp_factor(),
            in_color_space=int(self.jpeg_color_space),
            in_components=self.num_components,
            qt=qt,
            quality=quality,
            quant_tbl_no=quant_tbl_no,
            num_markers=self.num_markers,
            marker_types=self.c_marker_types(),
            marker_lengths=self.c_marker_lengths(),
            markers=self.c_markers(),
            flags=flags,
        )
        # clean up temporary file
        os.remove(tmp.name)

    @property
    def Y(self) -> np.ndarray:
        """Luminance tensor getter."""
        if self._Y is None:
            self.load()
        return self._Y

    @Y.setter
    def Y(self, Y: np.ndarray):
        """Luminance tensor setter."""
        self._Y = Y

    @property
    def Cb(self) -> np.ndarray:
        """Chrominance blue-difference tensor getter."""
        if self.has_chrominance and self._Cb is None:
            self.load()
        return self._Cb

    @Cb.setter
    def Cb(self, Cb: np.ndarray):
        """Chrominance blue-difference tensor setter."""
        self._Cb = Cb

    @property
    def Cr(self) -> np.ndarray:
        """Chrominance red-difference tensor getter."""
        if self.has_chrominance and self._Cr is None:
            self.load()
        return self._Cr

    @Cr.setter
    def Cr(self, Cr: np.ndarray):
        """Chrominance red-difference tensor setter."""
        self._Cr = Cr

    @property
    def K(self) -> np.ndarray:
        """Black tensor getter."""
        if self.has_black and self._K is None:
            self.load()
        return self._K

    @K.setter
    def K(self, K: np.ndarray):
        """Black tensor setter."""
        self._K = K

    @property
    def qt(self) -> np.ndarray:
        """Quantization table getter."""
        if self._qt is None:
            self.load()
        return self._qt

    @qt.setter
    def qt(self, qt: np.ndarray):
        """Quantization table setter."""
        self._qt = qt

    @property
    def quant_tbl_no(self) -> List:
        """Getter of assignment of quantization tables to components"""
        if self._quant_tbl_no is None:
            self.load()
        return self._quant_tbl_no

    @quant_tbl_no.setter
    def quant_tbl_no(self, quant_tbl_no: List):
        """Setter of assignment of quantization tables to components"""
        self._quant_tbl_no = quant_tbl_no

    def get_component_qt(self, idx: int) -> np.ndarray:
        """Getter of the quantization table of a component, dependent on assignment.

        :param idx: Component index.
        :type idx: int
        :return: quantization table of the component
        :rtype: np.ndarray

        :Examples:

        To retreive a component quantization table, simply type

        >>> jpeg = jpeglib.read_dct("input.jpeg")
        >>> # retrieval of component QTs
        >>> qtY = jpeg.get_component_qt(0)
        >>> qtCb = jpeg.get_component_qt(1)
        >>> qtCr = jpeg.get_component_qt(2)

        Usual assignment is Y having qt[0] and Cb and Cr sharing qt[1].
        However it may differ.

        >>> # retrieval of QTs at index
        >>> qt0 = jpeg.qt[0] # usually qtY
        >>> qt1 = jpeg.qt[1] # usually qtCb, qtCr (shared)

        E.g. ffmpeg uses a single quantization matrix for all three channels.
        """
        return self.qt[self.quant_tbl_no[idx]]

    def free(self):
        """Free the allocated tensors."""
        del self._Y
        del self._Cb
        del self._Cr
        del self._K
        del self._qt


class DCTJPEGio:
    pass


@dataclasses.dataclass
class ComponentInfo:
    """ComponentInfo from jpegio, intended for reading only"""
    component_id: int
    """component id"""
    jpeg: DCTJPEGio
    """DCTJPEGio instance"""

    @property
    def h_samp_factor(self):
        return self.jpeg.samp_factor[self.component_id, 0]

    @property
    def v_samp_factor(self):
        return self.jpeg.samp_factor[self.component_id, 1]

    @property
    def quant_tbl_no(self):
        return self.jpeg.quant_tbl_no[self.component_id]

    @property
    def ac_tbl_no(self):
        raise NotImplementedError

    @property
    def dc_tbl_no(self):
        raise NotImplementedError

    @property
    def downsampled_height(self):
        return self.jpeg.height / self.v_samp_factor

    @property
    def downsampled_width(self):
        return self.jpeg.width / self.h_samp_factor

    @property
    def height_in_blocks(self):
        return self.jpeg.height_in_blocks(self.component_id)

    @property
    def width_in_blocks(self):
        return self.jpeg.width_in_blocks(self.component_id)


@dataclasses.dataclass
class DCTJPEGio(DCTJPEG):
    """Class for compatiblity with jpegio."""

    coef_arrays: List
    """DCT coefficient arrays in jpegio format"""
    quant_tables: List
    """quantization tables in jpegio format"""
    comp_info: List
    """component infos"""

    def __post_init__(self):
        self._jpeg_to_jpegio()

    def _jpeg_to_jpegio(self):
        # colored image
        if self.has_chrominance:
            self._coef_arrays = [
                self._convert_dct_jpegio(self.Y),
                self._convert_dct_jpegio(self.Cb),
                self._convert_dct_jpegio(self.Cr)
            ]
            self._quant_tables = [
                self.get_component_qt(0).astype(np.int32),
                self.get_component_qt(1).astype(np.int32)
            ]
        # grayscale
        else:
            self._coef_arrays = [self._convert_dct_jpegio(self.Y)]
            self._quant_tables = [self.get_component_qt(0).astype(np.int32)]
        # component infos
        self.comp_info = [
            ComponentInfo(component_id=i, jpeg=self)
            for i in range(self.num_components)
        ]

    def _jpegio_to_jpeg(self):
        self.Y = self._convert_jpegio_dct(self.coef_arrays[0])
        if self.has_chrominance:
            self.Cb = self._convert_jpegio_dct(self.coef_arrays[1])
            self.Cr = self._convert_jpegio_dct(self.coef_arrays[2])
        if self.has_chrominance:
            self.qt = np.array([
                self.quant_tables[0],
                self.quant_tables[1]
            ], dtype=np.uint16)
            self.quant_tbl_no = np.array([0, 1, 1])
        else:
            self.qt = np.array([
                self.quant_tables[0]
            ], dtype=np.uint16)
            self.quant_tbl_no = np.array([0])
        # TODO: component info

    def _convert_dct_jpegio(self, dct: np.ndarray) -> np.ndarray:
        # From jpeglib's 4D tensor to jpegio 2D
        num_vertical_blocks, num_horizontal_blocks = dct.shape[:2]

        return (
            dct
            .transpose((0, 2, 1, 3))
            .reshape((num_vertical_blocks * 8, num_horizontal_blocks * 8))
        ).astype(np.int32)

    def _convert_jpegio_dct(self, dct: np.ndarray) -> np.ndarray:
        # From jpegio 2D to jpeglib 4D
        assert dct.shape[0] % 8 == 0
        assert dct.shape[1] % 8 == 0

        num_vertical_blocks = dct.shape[0] // 8
        num_horizontal_blocks = dct.shape[1] // 8

        return (
            dct
            .reshape((num_vertical_blocks, 8, num_horizontal_blocks, 8))
            .transpose((0, 2, 1, 3))
        ).astype(np.int16)

    @property
    def coef_arrays(self) -> List:
        """Convertor of DCT coefficients to jpegio format."""
        return self._coef_arrays

    @coef_arrays.setter
    def coef_arrays(self, coef_arrays: List):
        """Setter of coefficient arrays in jpegio format."""
        self._coef_arrays = coef_arrays

    @property
    def quant_tables(self) -> List:
        """Convertor of quantization tables to jpegio format."""
        return self._quant_tables

    @quant_tables.setter
    def quant_tables(self, quant_tables: List):
        """Setter of quantization tables in jpegio format."""
        self._quant_tables = quant_tables

    @property
    def comp_info(self) -> List:
        """Convertor of component infos to jpegio format."""
        return self._comp_info

    @comp_info.setter
    def comp_info(self, comp_info: List):
        """Setter of component infos in jpegio format."""
        self._comp_info = comp_info

    def write(self, fpath: str, flags: int = -1, quality: int = -1):
        """Function to write DCT coefficients in jpegio format to JPEG file.

        Does not perform JPEG compression, writing DCT is lossless.

        :param fpath: Destination file name. If not given, source file is overwritten.
        :type fpath: str, optional
        :param flags: Flags. For backward compatibility, does not make a difference.
        :type flags: int
        :param quality: Compression quality, between 0 and 100. Special value -1 stands for using qt inside the instance or keeping libjpeg default.
        :type quality: int, optional

        :Example:

        >>> jpeg = jpeglib.read_dct("input.jpeg")
        >>> jpeg = jpeglib.to_jpegio(jpeg)
        >>> jpeg.write("output.jpeg", quality=92)
        """  # noqa: E501
        # pass data from jpegio to jpeg
        self._jpegio_to_jpeg()
        # write
        self.write_dct(path=fpath, quality=quality)


def to_jpegio(jpeg: DCTJPEG) -> DCTJPEGio:
    """Convertor of object of :class:`DCTJPEG` to :class:`DCTJPEGio`.

    When :class:`DCTJPEG` is converted to :class:`DCTJPEGio`,
    the data objects :class:`DCTJPEG` are not updated until writing.
    This behavior will be changed in the future.

    :param jpeg: JPEG object to convert
    :type jpeg: :class:`DCTJPEG`
    :return: converted JPEG object
    :rtype: :class:`DCTJPEGio`

    :Example:

    >>> jpeg = jpeglib.read_dct("input.jpeg")
    >>> jpeg = jpeglib.to_jpegio(jpeg)
    >>> jpeg.coef_arrays; jpeg.quant_tables
    """
    vals = {field.name: getattr(jpeg, field.name) for field in dataclasses.fields(jpeg)}
    return DCTJPEGio(**vals)
