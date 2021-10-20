"""
Module containing conversion of C-structures into numpy and back and memory management.
"""

import ctypes
import logging
import numpy as np
import tempfile
import warnings

from ._bind import CJpegLib
from ._timer import Timer

class JPEG:
    cjpeglib = CJpegLib
    def __init__(self, srcfile):
        """Object constructor.
        
        :param srcfile: Source file.
        :type srcfile: str
        :raises [IOError]: When file does not exist or can't be read.
        """
        # set filename
        self.srcfile = srcfile
        # allocate resources
        self._dct_dims = (ctypes.c_int*6)()
        self._dims = (ctypes.c_int*3)()
        self._num_components = (ctypes.c_int*1)()
        self._color_space = (ctypes.c_int*1)()
        # get image info
        self._read_info()   
        # allocate
        self._im_spatial = self._allocate_spatial()
        self._im_dct = self._allocate_dct()
        self._im_qt = ((ctypes.c_short*64)*self.dct_channels)()
        self._samp_factor = ((ctypes.c_int*2)*3)()
    
    def read_dct(self):
        """Reads the file DCT.
        
        :return: Luminance tensor, chrominance tensor and quantization tables.
        :rtype: 3-tuple of np.ndarrays

        :Example:

        >>> import jpeglib
        >>> im = jpeglib.JPEG("input.jpeg")
        >>> Y,CbCr,qt = im.read_dct()
        """
        #t = Timer('reading %s DCT', self.srcfile) # log execution time
        # execute
        Y,CbCr,qt = self._read_dct(self.srcfile)
        # result
        return Y,CbCr,qt

    def write_dct(self, dstfile, Y=None, CbCr=None): # qt, 
        """Writes DCT coefficients to a file.
        
        :param dstfile: Destination file.
        :type dstfile: str
        :param Y: Luminance DCT.
        :type Y: np.ndarray
        :param CbCr: Chrominance DCT.
        :type CbCr: np.ndarray

        :Example:

        >>> import jpeglib
        >>> im = jpeglib.JPEG("input.jpeg")
        >>> Y,CbCr,qt = im.read_dct()
        >>> im.write_dct("output.jpeg", Y, CbCr)
        """
        #t = Timer('writing %s DCT', dstfile) # log execution time
        # execute
        self._write_dct(dstfile, Y, CbCr)
        self._im_dct = None # free DCT buffer
        self._im_spatial = None # free DCT buffer

    # https://bitmiracle.github.io/libjpeg.net/help/api/BitMiracle.LibJpeg.Classic.J_COLOR_SPACE.html
    J_COLOR_SPACE = {
        None: (-1,-1), # default
        "JCS_UNKNOWN":(0,0), # Unspecified color space
        "JCS_GRAYSCALE":(1,1), # Monochrome
        "JCS_RGB":(2,3), # Standard RGB
        "JCS_YCbCr":(3,3), # YCbCr or YUB, standard YCC
        "JCS_CMYK":(4,4), # CMYK
        "JCS_YCCK":(5,4) # YCbCrK
    }
    J_DITHER_MODE = {
        None: -1,
        "JDITHER_NONE": 0,
        "JDITHER_ORDERED": 1,
        "JDITHER_FS": 2
    }
    J_DCT_METHOD = {
        None: -1,
        "JDCT_ISLOW": 0, # slow but accurate integer algorithm
        "JDCT_IFAST": 1, # faster, less accurate integer method
        "JDCT_FLOAT": 2, # floating-point method
    }

    def read_spatial(self, out_color_space=None, dither_mode=None, dct_method=None, flags=[]):
        """Decompresses the file into the spatial domain. 
        
        :param out_color_space: Output color space. Must be key of J_COLOR_SPACE.
        :type out_color_space: str, optional
        :param dither_mode: Dither mode. Must be key of :class:`jpeg.JPEG.J_DITHER_MODE`. Using default from libjpeg by default.
        :type dither_mode: str, optional
        :param dct_method: DCT method. Must be key of :class:`jpeg.JPEG.J_DCT_METHOD`. Using default from libjpeg by default.
        :type dct_method: str, optional
        :param flags: Bool decompression parameters as str to set to be true. Using default from libjpeg by default.
        :type flags: list, optional
        :return: Spatial representation of the source image.
        :rtype: np.ndarray 
        """
        #t = Timer('reading %s spatial', self.srcfile) # log execution time
        # execute
        spatial = self._read_spatial(self.srcfile, out_color_space, dither_mode, dct_method, flags)
        self._im_dct = None # free DCT buffer
        # result
        return spatial

    def write_spatial(self, dstfile, data=None, in_color_space=None, dct_method="JDCT_ISLOW",
                      samp_factor=None, quality=100, smoothing_factor=None, flags=[]):
        """Writes spatial image representation (i.e. RGB) to a file.
        
        :param dstfile: Destination file name.
        :type dstfile: str
        :param data: Numpy array with spatial data.
        :type data: np.darray, optional
        :param in_color_space: Input color space. Must be key of :class:`jpeg.JPEG.J_COLOR_SPACE`. According to source by default.
        :type in_color_space: str, optional
        :param dct_method: DCT method. Must be key of :class:`jpeg.JPEG.J_DCT_METHOD`. Using default from libjpeg by default.
        :type dct_method: str, optional
        :param samp_factor: Sampling factor. None, tuple of three ints or tuples of two ints. According to source by default.
        :type samp_factor: str, optional
        :param quality: Compression quality, between 0 and 100. Defaultly 100 (full quality).
        :type quality: int, optional
        :param smoothing_factor: Smoothing factor, between 0 and 100. Using default from libjpeg by default.
        :type smoothing_factor: int, optional
        :param flags: Bool decompression parameters as str to set to be true. Using default from libjpeg by default.
        :type flags: list, optional
        """
        #t = Timer('writing %s spatial', self.srcfile) # log execution time
        # execute
        self._write_spatial(dstfile, data, in_color_space, dct_method, samp_factor, quality, smoothing_factor, flags)
        self._im_dct = None # free DCT buffer
        self._im_spatial = None # free spatial buffer

    def to_spatial(self, Y=None, CbCr=None, **kw): #, qt=None):
        """Converts DCT representation to spatial.
        
        Performs decompression from DCT coefficients to spatial representation.
        Uses temporary file to save and read from as libjpeg does not have a direct handler.
        
        :param Y: Luminance DCT.
        :type Y: np.ndarray
        :param CbCr: Chrominance DCT.
        :type CbCr: np.ndarray
        :return: Spatial representation of DCT coefficients
        :rtype: np.ndarray 
        """
        #t = Timer('DCT-RGB conversion')
        if (Y is None or CbCr is None) and self._im_dct is None:
            raise RuntimeError("Call read_dct() before calling to_spatial() or specify Y and CbCr.")
        with tempfile.NamedTemporaryFile() as tmp:
            self.write_dct(tmp.name, Y=Y, CbCr=CbCr)
            data = self._read_spatial(tmp.name, **kw)
        self._im_dct = None
        return data

    def _read_info(self):
        # get information
        self.cjpeglib.read_jpeg_info(
            srcfile = self.srcfile,
            dct_dims = self._dct_dims,
            image_dims = self._dims,
            num_components = self._num_components,
            jpeg_color_space = self._color_space
        )
        # parse
        self.dct_channels = 3
        self.channels = self._num_components[0]
        self.dct_shape = np.array([self._dct_dims[i] for i in range(6)], int)\
            .reshape(self.dct_channels, 2)
        self.shape = np.array([self._dims[0], self._dims[1]])
        self.color_space = [k for k,v in self.J_COLOR_SPACE.items() if v[0] == self._color_space[0]][0]

    
    def _read_dct(self, srcfile):
        # allocate
        if self._im_dct is None:
            self._im_dct = self._allocate_dct()
        # reading
        self.cjpeglib.read_jpeg_dct(srcfile, self._im_dct, self._im_qt)
        # align qt
        qt = np.ctypeslib.as_array(self._im_qt)
        qt = qt.reshape((*qt.shape[:-1],8,8))
        qt[2,:] = qt[1,:]
        # align lumo
        Y = np.ctypeslib.as_array(self._im_dct[:1])
        Y = Y.reshape((*Y.shape[:-1],8,8))
        # align chroma 
        CbCr = np.ctypeslib.as_array(self._im_dct[1:])
        CbCr = CbCr[:,:self.dct_shape[1][0],:self.dct_shape[1][1]]
        CbCr = CbCr.reshape((*CbCr.shape[:-1],8,8))
        # finish
        return Y,CbCr,qt

    def _write_dct(self, dstfile, Y=None, CbCr=None): #, qt=None)
        # TODO: remove copying from source file
        # allocate
        if self._im_dct is None:
            self._im_dct = self._allocate_dct()
        # reading
        self.cjpeglib.read_jpeg_dct(self.srcfile, self._im_dct, self._im_qt)

        # TODO: changed in QT
        # align lumo
        if Y is not None:
            Y = Y.reshape((*Y.shape[:-2],64))
            self._im_dct[:1] = np.ctypeslib.as_ctypes(Y)
        # align chroma
        if CbCr is not None:
            CbCr = CbCr.reshape((*CbCr.shape[:-2],64))
            _CbCr = np.zeros((2, self.dct_shape[0][0], self.dct_shape[0][1], 64), np.short)
            _CbCr[:,:int(self.dct_shape[1][0]),:self.dct_shape[1][1]] = CbCr
            self._im_dct[1:] = np.ctypeslib.as_ctypes(_CbCr)

        # write
        self.cjpeglib.write_jpeg_dct(self.srcfile, dstfile, self._im_dct)


    def _read_spatial(self, srcfile, out_color_space, dither_mode, dct_method, flags):
        # parameters
        #print("Color spaces:", self.color_space, out_color_space)
        if out_color_space is None:
            out_color_space = self.color_space
        self.color_space = out_color_space
        color_space,channels = self.J_COLOR_SPACE[out_color_space]
        dither_mode = self.J_DITHER_MODE[dither_mode]
        dct_method = self.J_DCT_METHOD[dct_method]
        # allocate
        if self._im_spatial is None or channels != self.channels:
            self.channels = channels
            self._im_spatial = self._allocate_spatial()
        else: self.channels = channels
        
        #print(f"I read image with {self.channels} channels in colorspace {color_space}.")
        self.cjpeglib.read_jpeg_spatial(
            srcfile = srcfile,
            rgb = self._im_spatial,
            out_color_space = color_space,
            dither_mode = dither_mode,
            dct_method = dct_method,
            samp_factor = self._samp_factor,
            flags = flags
        )
        # align rgb
        data = np.ctypeslib.as_array(self._im_spatial).astype(np.ubyte)
        data = data.reshape(data.shape[2],-1,self.channels)
        # finish
        return data
    
    def _write_spatial(self, dstfile, data, in_color_space, dct_method, samp_factor, quality, smoothing_factor, flags):
        """"""
        # parameters
        if in_color_space is None:
            in_color_space = self.color_space
        in_color_space,channels = self.J_COLOR_SPACE[in_color_space]
        dct_method = self.J_DCT_METHOD[dct_method]
        self._samp_factor = self._parse_samp_factor(samp_factor)

        
        if quality is None: # not specified
            qt = None#self._im_qt
        else:
            # numeric quality
            try:
                quality = int(quality)
                qt = None
            # quantization table
            except:
                qt = np.array(quality).reshape(*quality.shape[:-2],64)
                qt = self._im_qt = np.ctypeslib.as_ctypes(quality)
                quality = None
        # spatial buffer
        if data is not None:
            data = data.reshape(self.channels,data.shape[1],-1)
            self._im_spatial = np.ctypeslib.as_ctypes(data)
            self._dims = (ctypes.c_int * 3)(data.shape[1], data.shape[2], data.shape[0])
        elif self._im_spatial is None:
            warnings.warn('Writing unsuccessful, call read_spatial() before calling write_spatial() or specify data parameter.', RuntimeWarning)
            return

        self.cjpeglib.write_jpeg_spatial(
            srcfile = self.srcfile,
            dstfile = dstfile,
            rgb = self._im_spatial,
            image_dims = self._dims,
            in_color_space = in_color_space,
            in_components = self.channels,
            dct_method = dct_method,
            samp_factor = self._samp_factor,
            quality = quality,
            qt = qt,
            smoothing_factor = smoothing_factor,
            flags = flags
        )

    def print_params(self): self.cjpeglib.print_jpeg_params(self.srcfile)
    def __enter__(self):
        """Method for using ``with`` statement together with :class:`JPEG`."""
        return self
    def __exit__(self, exception_type, exception_val, trace):
        """Method for using ``with`` statement together with :class:`JPEG`."""
        self.close()
    def close(self):
        """Closes the object. Defined for interface compatibility with PIL."""
        pass
    def _allocate_dct(self):
        return ((((ctypes.c_short * 64) 
                                  * self.dct_shape[0][1])
                                  * self.dct_shape[0][0])
                                  * self.dct_channels)()
    def _allocate_spatial(self):
        return (((ctypes.c_ubyte * self.shape[1])
                                 * self.shape[0])
                                 * self.channels)()
    def _parse_samp_factor(self, samp_factor):
        if samp_factor is not None:
            samp_factor = list(samp_factor)
            for i,f in enumerate(samp_factor):
                if isinstance(f,int):
                    self._samp_factor[i][0] = f
                    self._samp_factor[i][1] = f
                else:
                    self._samp_factor[i][0] = f[0]
                    self._samp_factor[i][1] = f[1]
        return self._samp_factor





__all__ = ["JPEG"]