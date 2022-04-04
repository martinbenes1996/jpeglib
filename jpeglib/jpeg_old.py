"""
Module containing conversion of C-structures into numpy and back and memory management.
"""

import ctypes
import logging
import math
import numpy as np
import tempfile
import warnings

from ._bind import CJpegLib
from ._timer import Timer
from ._notations import Jab_to_factors

class JPEG:
    cjpeglib = CJpegLib
    def __init__(self, srcfile=None):
        """Constructs the :class:`JPEG` object.
        
        :param srcfile: Path to a source file in JPEG format. An implicit destination for writing.
        :type srcfile: str
        :raises [IOError]: When file does not exist or can't be read.
        
        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        """
        # set filename
        self.srcfile = srcfile
        # allocate resources
        self._dct_dims = (ctypes.c_int*6)()
        self._dims = (ctypes.c_int*3)()
        self._num_components = (ctypes.c_int*1)()
        self._im_qt = ((ctypes.c_short*64)*(4))()
        self._samp_factor = None # self._parse_samp_factor()
        self.color_space = 'JCS_RGB'
        #self.color_space = [k for k,v in self.J_COLOR_SPACE.items() if v[0] == self._color_space[0]][0]
        # get image info
        if srcfile is not None:
            try:
                self._read_info()
                self._allocate() # allocate
            except:
                self._im_spatial = None
                self._im_colormap = None
                self._im_dct = None
        else:
            self._im_spatial = None
            self._im_colormap = None
            self._im_dct = None

    def read_dct(self, quantized=False):
        """Reads the DCT coefficients and quantization tables of the source file.

        In the return values (when format is 'block'), Y is DCT luminance tensor of shape (1,W/8,H/8,8,8),
        CbCr is DCT chrominance tensor of shape (2,W/8/sw,H/8/sh,8,8), both not-quantized by default,
        where sw, sh are corresponsing chrominance scaling factors.
        qt is a tensor with quantization tables of shape (2,8,8).
        
        When format is 'jpegio', DCT tensors have shapes (W,H) and (W/sw, H/sh).
        Check the examples below.

        :param quantized: Indicates whether the output DCT coefficients are quantized or not, False by default.
        :type quantized: bool, optional
        :param format: Format of output, one of 'block', 'jpegio'.
        :type format: str, optional
        :return: lumo DCT Y, chroma DCT CbCr, quantization table qt
        :rtype: tuple
        :raises [IOError]: When source file was not given in constructor.

        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        >>> Y,CbCr,qt = im.read_dct()

        Get DCT coefficients in shape (W/8, H/8, 8, 8) with

        >>> Y[0]    # Y
        >>> CbCr[0] # Cb
        >>> CbCr[1] # Cr

        Get quantization tables in shape (8,8) with

        >>> qt[0] # Y
        >>> qt[1] # Cb
        >>> qt[1] # Cr
        
        Get DCT coefficients in shape (W, H) with
        
        >>> Y,CbCr,qt = im.read_dct(format='jpegio')
        """
        # execute
        Y,CbCr,qt = self._read_dct(self.srcfile)
        # quantization
        if quantized: Y,CbCr = Y*qt[0],CbCr*qt[1]
        # format
        if format == 'block': pass
        elif format == 'jpegio':
            Y = (Y
                .transpose((0,2,4,1,3))
                .reshape((Y.shape[2]*Y.shape[4], Y.shape[1]*Y.shape[3]))
            )
            CbCr = (CbCr
                    .transpose((0,2,4,1,3))
                    .reshape((Y.shape[2]*Y.shape[4], Y.shape[1]*Y.shape[3]))
            )
        # result
        return Y,CbCr,qt

    def write_dct(self, Y, CbCr=None, dstfile=None, qt=None, quantized=False, format='block', in_color_space=None, samp_factor=None):
        """Writes DCT coefficients to a file.
        
        The shape and arrangement of the parameters Y and CbCr is the same
        as what is returned from :meth:`jpeg.JPEG.read_dct`.
        Y is DCT luminance tensor of shape (1,W/8,H/8,8,8), CbCr is DCT chrominance tensor of shape (2,W/8,H/8,8,8),
        both unquantized by default. qt is either a tensor with quantization tables of shape (2,8,8) or a quality scalar integer.

        :param Y: Lumo DCT.
        :type Y: numpy.ndarray
        :param CbCr: Chrominance DCT.
        :type CbCr: numpy.ndarray, optional
        :param dstfile: Destination file. If not given, source file is overwritten.
        :type dstfile: str, optional
        :param qt: Quantization tables or quality.
        :type qt: numpy.ndarray | int, optional
        :param quantized: Indicates whether the input DCT coefficients are quantized or unquantized, False by default.
        :type quantized: bool, optional 
        :param in_color_space: Input color space. Must be key of :class:`jpeg.JPEG.J_COLOR_SPACE`. According to source by default.
        :type in_color_space: str, optional
        :param samp_factor: Sampling factor. None, or list of 3 tuples of 2 ints. According to source by default. Read more at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#jpeg-sampling-factor>`_.
        :type samp_factor: list, optional

        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        >>> Y,CbCr,qt = im.read_dct()
        >>> # work with DCT coefficients
        >>> # ...
        >>> im.write_dct("output.jpeg", Y, CbCr, qt)

        You can specify the quality scalar such as

        >>> im.write_dct("output.jpeg", Y, CbCr, 75)

        If neither quality nor quantization tables are specified,
        library uses the quantization table from the source.
        """
        # choose destination
        if dstfile is None:
            if self.srcfile is None:
                raise Exception("destination not specified")
            dstfile = self.srcfile
        # execute
        self._write_dct(dstfile, Y, CbCr, qt, quantized, in_color_space, samp_factor)
        # free buffers
        self._im_dct = None
        self._im_spatial = None

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

    def read_spatial(self, out_color_space=None, dither_mode=None, dct_method=None, colormap=None, flags=[]):
        """Decompresses the file into the spatial domain. 
        
        :param out_color_space: Output color space. Must be key of J_COLOR_SPACE.
        :type out_color_space: str, optional
        :param dither_mode: Dither mode. Must be key of :class:`jpeg.JPEG.J_DITHER_MODE`. Using default from libjpeg by default.
        :type dither_mode: str, optional
        :param dct_method: DCT method. Must be key of :class:`jpeg.JPEG.J_DCT_METHOD`. Using default from libjpeg by default.
        :type dct_method: str, optional
        :param colormap: Colormap to use for color quantization.
        :type colormap: np.array, optional
        :param flags: Bool decompression parameters as str to set to be true. Using default from libjpeg by default.
        :type flags: list, optional
        :return: Spatial representation of the source image.
        :rtype: numpy.ndarray

        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        >>> rgb = im.read_spatial(out_color_space="JCS_RGB")
        """
        # execute
        spatial = self._read_spatial(self.srcfile, out_color_space, dither_mode, dct_method, colormap, flags)
        # free buffers
        self._im_dct = None
        # result
        return spatial

    def write_spatial(self, data, dstfile=None, in_color_space=None, dct_method="JDCT_ISLOW",
                      samp_factor=None, qt=-1, smoothing_factor=None, flags=[]):
        """Writes spatial image representation (i.e. RGB) to a file.
        
        :param data: Numpy array with spatial data.
        :type data: numpy.ndarray
        :param dstfile: Destination file name. If not given, source file is overwritten.
        :type dstfile: str, optional
        :param in_color_space: Input color space. Must be key of :class:`jpeg.JPEG.J_COLOR_SPACE`. According to source by default.
        :type in_color_space: str, optional
        :param dct_method: DCT method. Must be key of :class:`jpeg.JPEG.J_DCT_METHOD`. Using default from libjpeg by default.
        :type dct_method: str, optional
        :param samp_factor: Sampling factor. None, or list of 3 tuples of 2 ints. According to source by default. Read more at `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#jpeg-sampling-factor>`_.
        :type samp_factor: list, optional
        :param qt: Compression quality, between 0 and 100 or a tensor with quantization tables. Defaultly -1 (default factor kept).
        :type qt: int | numpy.ndarray, optional
        :param smoothing_factor: Smoothing factor, between 0 and 100. Using default from libjpeg by default.
        :type smoothing_factor: int, optional
        :param flags: Bool decompression parameters as str to set to be true. Using default from libjpeg by default.
        :type flags: list, optional

        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        >>> grayscale = im.read_spatial(out_color_space="JCS_GRAYSCALE")
        >>> # work with the data
        >>> # ...
        >>> im.write_spatial(grayscale)
        """
        # choose destination
        if dstfile is None:
            if self.srcfile is None:
                raise Exception("destination not specified")
            dstfile = self.srcfile
        # execute
        self._write_spatial(dstfile, data, in_color_space, dct_method, samp_factor, qt, smoothing_factor, flags)
        # free buffers
        self._im_dct = None
        self._im_spatial = None

    def source_color_space(self):
        """Returns the color space of the source file."""
        return self.color_space

    def to_spatial(self, Y=None, CbCr=None, qt=None, quantized=False, **kw): #, qt=None):
        """Converts DCT representation to spatial.
        
        Performs decompression from DCT coefficients to spatial representation.
        Uses temporary file to save and read from as libjpeg does not have a direct handler.
        
        :param Y: Lumo DCT.
        :type Y: numpy.ndarray, optional
        :param CbCr: Chrominance DCT.
        :type CbCr: numpy.ndarray, optional
        :param qt: Quantization tables or quality.
        :type qt: numpy.ndarray | int, optional
        :param quantized: Indicates whether the input DCT coefficients are quantized or unquantized, False by default.
        :type quantized: bool, optional
        :return: Spatial representation of DCT coefficients
        :rtype: numpy.ndarray

        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        >>> Y,CbCr,qt = im.read_dct()
        >>> spatial = im.to_spatial(Y, CbCr, out_color_space="JCS_RGB")
        """
        # TODO: grayscale
        if (Y is None or CbCr is None) and self._im_dct is None:
            raise RuntimeError("Call read_dct() before calling to_spatial() or specify Y and CbCr.")
        with tempfile.NamedTemporaryFile() as tmp:
            self.write_dct(tmp.name, Y=Y, CbCr=CbCr, qt=qt, quantized=quantized)
            with JPEG(tmp.name) as im:
                data = im.read_spatial(**kw)
        self._im_dct = None
        return data

    def _read_info(self):
        # get information
        _color_space = (ctypes.c_int*1)()
        self.cjpeglib.read_jpeg_info(
            srcfile = self.srcfile,
            dct_dims = self._dct_dims,
            image_dims = self._dims,
            num_components = self._num_components,
            samp_factor = self._samp_factor,
            jpeg_color_space = _color_space
        )
        # parse jpeg information
        self.channels = self._num_components[0] # number of color components (can be assumed 3?)
        self.dct_channels = self._num_components[0]
        self.dct_shape = np.array([self._dct_dims[i] for i in range(6)], int)\
            .reshape(self.dct_channels, 2)
        self.shape = np.array([self._dims[0], self._dims[1]])
        self.color_space = [k for k,v in self.J_COLOR_SPACE.items() if v[0] == _color_space[0]][0]

    def _allocate(self):
        self._im_spatial,self._im_colormap = self._allocate_spatial()
        self._im_dct = self._allocate_dct()

    def _initialize_info(self, data=None, Y=None, CbCr=None):
        if data is not None:
            self.dct_channels = self.channels = data.shape[2]
            self.shape = np.array(data.shape[:2])
        else:
            self.dct_channels = self.channels = 3
            self.shape = np.array([0,0])
        if Y is not None:
            self._dct_dims[0] = Y.shape[1]
            self._dct_dims[1] = Y.shape[2]
            if CbCr is not None:
                self._dct_dims[2] = CbCr.shape[1]
                self._dct_dims[3] = CbCr.shape[2]
                self._dct_dims[4] = CbCr.shape[1]
                self._dct_dims[5] = CbCr.shape[2]
                self.shape = np.array([self._dct_dims[1], self._dct_dims[0]]) * 8
                self.dct_channels = 3
            else:
                self.shape = np.array([self._dct_dims[1], self._dct_dims[0]]) * 8
                self.dct_channels = 1
            self.channels = self.dct_channels
        else:
            for i in range(6):
                self._dct_dims[i] = math.ceil((data.shape[i % 2] / 8))
                #if self._samp_factor[i // 2][i % 2] != 0:
                #    self._dct_dims[i] = math.ceil(self._dct_dims[i])# / self._samp_factor[i // 2][i % 2])

        self.dct_shape = np.array([self._dct_dims[i] for i in range(2*self.dct_channels)], int)\
            .reshape(self.dct_channels, 2)
        self._dims = (ctypes.c_int * 3)(self.shape[1], self.shape[0], self.channels)

        self._allocate()
    
    def _read_dct(self, srcfile):
        """"""
        # check source file
        if srcfile is None: raise IOError("source file not given")
        try: self.dct_shape
        except: raise IOError("source file does not exist")
        # allocate
        if self._im_dct is None:
            self._im_dct = self._allocate_dct()
        
        # reading
        self.cjpeglib.read_jpeg_dct(
            srcfile = srcfile,
            dct     = self._im_dct,
            qt      = self._im_qt
        )
        
        # align qt
        qt = np.ctypeslib.as_array(self._im_qt)
        qt = qt.reshape((*qt.shape[:-1],8,8))
        # align lumo
        Y = np.ctypeslib.as_array(self._im_dct[:1])
        Y = Y.reshape((*Y.shape[:-1],8,8))
        # align chroma
        CbCr = np.ctypeslib.as_array(self._im_dct[1:])
        CbCr = CbCr[:,:self.dct_shape[1][0],:self.dct_shape[1][1]]
        CbCr = CbCr.reshape((*CbCr.shape[:-1],8,8))
        # finish
        return Y,CbCr,qt


    def _write_dct(self, dstfile, Y, CbCr=None, quality=None, quantized=False, in_color_space=None, samp_factor=None):
        """"""
        # initialize default
        #if self.srcfile is None: # if no srcfile, internals need to be initialized
        self._initialize_info(Y=Y, CbCr=CbCr)
        samp_factor = self._parse_samp_factor(samp_factor, Y, CbCr)
        # allocate DCT
        if self._im_dct is None: # could have been previously freed
            self._im_dct = self._allocate_dct()
        # quantization table and quality
        if quantized and quality is None:
            qt = np.ctypeslib.as_array(self._im_qt)
        qtable,quality = self._parse_quality(quality)
        if quantized:
            if qtable is not None:
                qt = np.ctypeslib.as_array(qtable)
            elif self._im_qt is not None:
                qt = np.ctypeslib.as_array(self._im_qt)
            else:
                raise IOError("can't dequantize with quantization table not given")
        
        # align lumo
        Y = Y.reshape((*Y.shape[:-2],64))
        if quantized:
            Y = (Y / qt[0]).astype(np.int16)
        self._im_dct[0] = np.ctypeslib.as_ctypes(Y[0])
        # align chroma
        if CbCr is not None: # not given e.g. for grayscale
            CbCr = CbCr.reshape((*CbCr.shape[:-2],64))
            if quantized:
                CbCr = (CbCr / qt[1]).astype(np.int16)
            _CbCr = np.zeros((2, self.dct_shape[0][0], self.dct_shape[0][1], 64), np.short)
            _CbCr[:,:int(self.dct_shape[1][0]),:int(self.dct_shape[1][1])] = CbCr
            self._im_dct[1:] = np.ctypeslib.as_ctypes(_CbCr)
        
        # colorspace and channels
        if in_color_space is None:
            if CbCr is None: in_color_space = 'JCS_GRAYSCALE'
            else: in_color_space = 'JCS_RGB'
            #in_color_space = self.color_space
        in_color_space,channels = self.J_COLOR_SPACE[in_color_space]
        
        
        # write
        #print("write_jpeg_dct:", self._dims[:], in_color_space, self.channels, [samp_factor[i] for i in range(3)])
        self.cjpeglib.write_jpeg_dct(
            srcfile        = self.srcfile,
            dstfile        = dstfile,
            dct            = self._im_dct,
            image_dims     = self._dims,
            in_color_space = in_color_space,
            in_components  = self.channels,
            samp_factor    = samp_factor,
            qt             = qtable,
            quality        = quality
        )


    def _read_spatial(self, srcfile, out_color_space, dither_mode, dct_method, colormap, flags):
        """"""
        # check source file
        if srcfile is None: raise IOError("source file not given")
        
        # color space and channels
        if out_color_space is None and self.color_space is not None:
            out_color_space = self.color_space
        color_space,channels = self.J_COLOR_SPACE[out_color_space]
        # dither mode and dct method
        dither_mode = self.J_DITHER_MODE[dither_mode]
        dct_method = self.J_DCT_METHOD[dct_method]
        
        if any(i in flags for i in ['QUANTIZE_COLORS','+QUANTIZE_COLORS','-QUANTIZE_COLORS']):
            raise NotImplementedError("color quantization not fully implemented")
        in_cmap = None
        if colormap is not None:
            raise NotImplementedError("color quantization not fully implemented")
            #if 'QUANTIZE_COLORS' not in flags and '+QUANTIZE_COLORS' not in flags:
            #    flags.append('+QUANTIZE_COLORS')
            #in_cmap = np.ascontiguousarray(colormap)
            #in_cmap = np.ctypeslib.as_ctypes(in_cmap.astype(np.ubyte))
        
        # allocate buffers
        if self._im_spatial is None or channels != self.channels:
            if channels > 0:
                self.channels = channels
            self._im_spatial,self._im_colormap = self._allocate_spatial()
        else: self.channels = channels # allocated correctly

        # call
        self.cjpeglib.read_jpeg_spatial(
            srcfile         = srcfile,
            rgb             = self._im_spatial,
            colormap        = self._im_colormap,
            in_colormap     = in_cmap,
            out_color_space = color_space,
            dither_mode     = dither_mode,
            dct_method      = dct_method,
            flags           = flags
        )
        
        # align rgb
        data = np.ctypeslib.as_array(self._im_spatial).astype(np.ubyte)
        
        # if 'QUANTIZE_COLORS' in flags and '+QUANTIZE_COLORS' in flags:
        #     data = data.reshape(self.channels,data.shape[2],-1)[0]
        #     # parse colormap
        #     colormap = np.ctypeslib.as_array(self._im_colormap)
        #     colormap = colormap.reshape(colormap.shape[1], self.channels)
        #     # index to color
        #     data = np.array([[colormap[i] for i in row] for row in np.array(data)])
        # else:
        #     #print(data.shape, self.channels)
        #     data = data.reshape(data.shape[2],-1,self.channels)
        # quantization
        #if 'QUANTIZE_COLORS' in flags:
        #    data = data[:,:,0]
        data = data.reshape(data.shape[2],-1,self.channels)
        # finish
        return data
    
    def _write_spatial(self, dstfile, data, in_color_space, dct_method, samp_factor, quality, smoothing_factor, flags):
        """"""
        # initialize default
        #if self.srcfile is None:
        #    self._initialize_info(data=data)
        self._initialize_info(data=data)
        self._samp_factor = self._parse_samp_factor(samp_factor)
        
        # parameters
        if in_color_space is None:
            if data.shape[2] == 1:
                in_color_space = 'JCS_GRAYSCALE'
            elif data.shape[2] == 3:
                in_color_space = 'JCS_RGB'
            else:
                raise Exception('unknown colorspace')
        #in_color_space = in_color_space if in_color_space is not None else self.color_space
        color_space,channels = self.J_COLOR_SPACE[in_color_space]
        dct_method = self.J_DCT_METHOD[dct_method]
        qt,quality = self._parse_quality(quality)
        
        # buffer
        if data is not None:
            data = data.reshape(self.channels,data.shape[1],-1)
            self._im_spatial = np.ctypeslib.as_ctypes(data)
            self._dims = (ctypes.c_int * 3)(data.shape[1], data.shape[2], data.shape[0])
        elif self._im_spatial is None:
            warnings.warn('Writing unsuccessful, call read_spatial() before calling write_spatial() or specify data parameter.', RuntimeWarning)
            return
        # call
        self.cjpeglib.write_jpeg_spatial(
            dstfile          = dstfile,
            rgb              = self._im_spatial,
            image_dims       = self._dims,
            in_color_space   = color_space,
            in_components    = self.channels,
            dct_method       = dct_method,
            samp_factor      = self._samp_factor,
            quality          = quality,
            qt               = qt,
            smoothing_factor = smoothing_factor,
            flags            = flags
        )

    def print_params(self): self.cjpeglib.print_jpeg_params(self.srcfile)
    def __enter__(self):
        """Method for using ``with`` statement together with :class:`JPEG`.
        
        :Example:
        
        >>> with jpeglib.JPEG("input.jpeg") as im:
        >>>     Y,CbCr,qt = im.read_dct()
        """
        return self
    def __exit__(self, exception_type, exception_val, trace):
        """Method for using ``with`` statement together with :class:`JPEG`."""
        self.close()
    def close(self):
        """Closes the object. Defined for interface compatibility with PIL.
        
        :Example:

        >>> im = jpeglib.JPEG("input.jpeg")
        >>> # work with im
        >>> im.close()
        """
        pass
    def _allocate_dct(self):
        return ((((ctypes.c_short * 64) 
                                  * self.dct_shape[0][1])
                                  * self.dct_shape[0][0])
                                  * self.dct_channels)()
    def _allocate_spatial(self):
        return ((((ctypes.c_ubyte * self.shape[1])
                                 * self.shape[0])
                                 * self.channels)(),
                ((ctypes.c_ubyte * 256) * self.channels)())
        
    def _parse_samp_factor(self, samp_factor=None, Y=None, CbCr=None):
        # sampling factor not given
        if samp_factor is None:
            # deduce sampling from shape of Y and CbCr
            if Y is not None and CbCr is not None:
                hs = [Y.shape[1], CbCr.shape[1], CbCr.shape[1]]
                ws = [Y.shape[2], CbCr.shape[2], CbCr.shape[2]]
                hmax,wmax = max(hs),max(ws)
                hs,ws = [h/hmax for h in hs],[w/wmax for w in ws]
                hmin,wmin = min(hs),min(ws)
                hs,ws = [round(h/hmin) for h in hs],[round(w/wmin) for w in ws]
                samp_factor = [[h,w] for h,w in zip(hs,ws)]
        # default samp factor
        #if samp_factor is None:
        #    samp_factor = [[2,2],[1,1],[1,1]]
        # process given sampling factor
        if samp_factor is not None:
            assert(len(samp_factor) == 3)
            # J:a:b notation
            if [len(s) for s in samp_factor] == [1,1,1]:
                samp_factor = Jab_to_factors(samp_factor)
        # check factors are valid
        if samp_factor is not None:
            samp_factor = np.array(samp_factor, dtype=np.int32)
            if np.sum(np.prod(samp_factor, axis=1)) > 10:
                raise Exception("linear combination of samp_factor greater than 10")
            if np.any(~np.isin(samp_factor,[1,2,3,4])):
                raise Exception("factors not in 1,2,3,4")
            # convert to ctypes
            self._samp_factor = np.ctypeslib.as_ctypes(samp_factor)
        return self._samp_factor
    
    def _parse_quality(self, quality):
        if quality is None: # not specified
            qt = self._im_qt
        else:
            # numeric quality
            try:
                quality = int(quality)
                qt = None
            # quantization table
            except:
                qt = np.array(quality, dtype=np.uint16)#.reshape(2,64)
                qt = self._im_qt = np.ctypeslib.as_ctypes(qt)
                quality = -1
        return qt,quality




__all__ = ["JPEG"]