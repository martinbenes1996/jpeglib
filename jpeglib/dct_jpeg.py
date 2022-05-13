
import ctypes
from dataclasses import dataclass,fields
import numpy as np
import tempfile
from ._bind import CJpegLib
from . import _jpeg

@dataclass
class DCTJPEG(_jpeg.JPEG):
    """JPEG instance to work with DCT domain."""
    
    Y: np.ndarray
    """luminance tensor"""
    Cb: np.ndarray
    """chrominance blue-difference tensor"""
    Cr: np.ndarray
    """chrominance red-difference tensor"""
    qt: np.ndarray
    """quantization tensor"""
    quant_tbl_no: np.ndarray
    """assignment of quantization tables to components, (0 Y, 1 Cb, 1Cr) by default"""
    
    # def is_read(self) -> bool:
    #     has_Y = self.Y is not None
    #     has_qt = self.qt is not None
    #     has_CbCr = self.Cb is not None and self.Cr is not None
    #     has_no_CbCr = self.Cb is None and self.Cr is None
    #     return (
    #         (has_Y and has_qt and has_no_CbCr) # grayscale
    #         or (has_Y and has_qt and has_CbCr) # color
    #     )
    
    def _alloc_dct_component(self, i:int):
        return (((ctypes.c_short * 64) * self.width_in_blocks(i)) * self.height_in_blocks(i))()
    
    def read_dct(self) -> tuple:
        """Function to allocate the buffer and read the image data.
        
        :return: luminance Y and chrominance Cb, Cr tensors and quantization table qt as Y,(Cb,Cr),qt
        :rtype: tuple
        
        :Example:
        
        >>> jpeg = jpeglib.read_dct("input.jpeg")
        >>> Y,(Cb,Cr),qt = jpeg.read_dct()
        
        Function is internally called when
        
        >>> jpeg = jpeglib,read_dct("input.jpeg")
        >>> jpeg.Y # jpeg.read_dct() called internally
        """
        # no content
        if self.content is None:
            return None,(None,None),None
        # write content into temporary file
        tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')
        tmp.write(self.content)
        tmp.flush()
        # allocate DCT components
        Y = self._alloc_dct_component(0)
        Cb,Cr = None,None
        if self.has_chrominance: # has chrominance
            Cb = self._alloc_dct_component(1)
            Cr = self._alloc_dct_component(2)
        qt = ((ctypes.c_short * 64) * 4)()
        _quant_tbl_no = (ctypes.c_ubyte*4)()
        # call
        CJpegLib.read_jpeg_dct(
            path            = self.path,
            srcfile         = tmp.name,
            Y               = Y,
            Cb              = Cb,
            Cr              = Cr,
            qt              = qt,
            quant_tbl_no    =_quant_tbl_no,
        )
        # close temporary file
        tmp.close()
        # process
        def process_component(comp):
            comp_np = np.ctypeslib.as_array(comp)
            return comp_np.reshape((*comp_np.shape[:-1],8,8))
        qt = process_component(qt)
        self.Y = process_component(Y)
        if self.has_chrominance:
            self.Cb = process_component(Cb)
            self.Cr = process_component(Cr)
        self.quant_tbl_no = np.array([_quant_tbl_no[i] for i in range(self.num_components)])
        # crop
        self.qt = qt[:self.num_components]
        # return
        return self.Y,(self.Cb,self.Cr),self.qt
    
    def write_dct(self, path:str = None, quality:int=-1):
        """Function to write DCT coefficients to JPEG file.
        
        Does not perform JPEG compression, writing DCT is lossless.
        
        :param path: Destination file name. If not given, source file is overwritten.
        :type path: str, optional
        :param quality: Compression quality, between 0 and 100. Special value -1 stands for using qt inside the instance or keeping libjpeg default.
        :type quality: int, optional
        
        :Example:
        
        >>> jpeg = jpeglib.read_spatial("input.jpeg")
        >>> jpeg.write_spatial("output.jpeg", quality=92)
        """
        tmp = tempfile.NamedTemporaryFile(suffix='jpeg')
        if self.content is not None:
            # write content into temporary file
            tmp.write(self.content)
            tmp.flush()
        # path
        dstfile = path if path is not None else self.path
        if dstfile is None:
            raise IOError('no destination file specified')
        # convert dct
        def process_component(comp):
            if comp is None:
                return None
            comp = comp.reshape((*comp.shape[:-2],64))
            return np.ctypeslib.as_ctypes(comp.astype(np.int16))
        qt = process_component(self.qt)
        Y = process_component(self.Y)
        Cb = process_component(self.Cb)
        Cr = process_component(self.Cr)
        # quality and quantization table
        assert(quality in set(range(-1,101)))
        if quality != -1:
            qt = None
            
        # call
        CJpegLib.write_jpeg_dct(
            srcfile         = tmp.name if self.content is not None else None,
            dstfile         = dstfile,
            Y               = Y,
            Cb              = Cb,
            Cr              = Cr,
            image_dims      = self.c_image_dims(),
            block_dims      = self.c_block_dims(),
            in_color_space  = self.jpeg_color_space.index,
            in_components   = self.num_components,
            qt              = qt,
            quality         = quality,
            num_markers     = self.num_markers,
            marker_types    = self.c_marker_types(),
            marker_lengths  = self.c_marker_lengths(),
            markers         = self.c_markers(),
        )
    
    @property
    def Y(self) -> np.ndarray:
        """Luminance tensor getter."""
        if self._Y is None:
            self.read_dct()
        return self._Y
    @Y.setter
    def Y(self, Y: np.ndarray):
        """Luminance tensor setter."""
        self._Y = Y
    
    @property
    def Cb(self) -> np.ndarray:
        """Chrominance blue-difference tensor getter."""
        if self.has_chrominance and self._Cb is None:
            self.read_dct()
        return self._Cb
    @Cb.setter
    def Cb(self, Cb: np.ndarray):
        """Chrominance blue-difference tensor setter."""
        self._Cb = Cb
    
    @property
    def Cr(self) -> np.ndarray:
        """Chrominance red-difference tensor getter."""
        if self.has_chrominance and self._Cr is None:
            self.read_dct()
        return self._Cr
    @Cr.setter
    def Cr(self, Cr: np.ndarray):
        """Chrominance red-difference tensor setter."""
        self._Cr = Cr
    
    @property
    def qt(self) -> np.ndarray:
        """Quantization table getter."""
        if self._qt is None:
            self.read_dct()
        return self._qt
    @qt.setter
    def qt(self, qt: np.ndarray):
        """Quantization table setter."""
        self._qt = qt
    @property
    def quant_tbl_no(self) -> list:
        """Getter of assignment of quantization tables to components"""
        if self._quant_tbl_no is None:
            self.read_dct()
        return self._quant_tbl_no
    @quant_tbl_no.setter
    def quant_tbl_no(self, quant_tbl_no:list):
        """Setter of assignment of quantization tables to components"""
        self._quant_tbl_no = quant_tbl_no
    
    def get_component_qt(self, idx:int) -> np.ndarray:
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
        
        Usual assignment is Y having qt[0] and Cb and Cr sharing qt[1]. However it may differ.
        
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
        del self._qt

        
@dataclass
class DCTJPEGio(DCTJPEG):
    """Class for compatiblity with jpegio."""
    
    coef_arrays: list
    """DCT coefficient arrays in jpegio format"""
    quant_tables: list
    """quantization tables in jpegio format"""
    
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
            self.quant_tbl_no = np.array([0,1,1])
        else:
            self.qt = np.array([
                self.quant_tables[0]
            ], dtype=np.uint16)
            self.quant_tbl_no = np.array([0])
    
    def _convert_dct_jpegio(self, dct:np.ndarray):
        return (dct
            .transpose((0,3,1,2))
            .reshape((dct.shape[0]*dct.shape[2], dct.shape[1]*dct.shape[3]))
        ).astype(np.int32)
    def _convert_jpegio_dct(self, dct:np.ndarray):
        return (dct
            .reshape((-1,8,dct.shape[1]//8,8))
            .transpose((0,2,3,1))
        ).astype(np.int16)
    
    @property
    def coef_arrays(self) -> list:
        """Convertor of DCT coefficients to jpegio format."""
        return self._coef_arrays

    @coef_arrays.setter
    def coef_arrays(self, coef_arrays:list):
        """Setter of coefficient arrays in jpegio format."""
        self._coef_arrays = coef_arrays
    
    @property
    def quant_tables(self) -> list:
        """Convertor of quantization tables to jpegio format."""
        return self._quant_tables
    @quant_tables.setter
    def quant_tables(self, quant_tables:list):
        """Setter of quantization tables in jpegio format."""
        self._quant_tables = quant_tables
    
    def write(self, fpath:str, flags:int=-1, quality:int=-1):
        """Function to write DCT coefficients in jpegio format to JPEG file.
    
        Does not perform JPEG compression, writing DCT is lossless.
    
        :param obj:
        :type obj: str
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
        """
        # pass data from jpegio to jpeg
        self._jpegio_to_jpeg()
        # write 
        self.write_dct(path=fpath, quality=quality)
        
def to_jpegio(jpeg: DCTJPEG):
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
    vals = {field.name: getattr(jpeg,field.name) for field in fields(jpeg)}
    return DCTJPEGio(**vals)
