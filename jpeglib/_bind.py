
import ctypes
import os
import pathlib
import re

from . import cjpeglib

class CJpegLib:

    @classmethod
    def jpeg_lib_version(cls):
        return cls.get().jpeg_lib_version()

    @classmethod
    def read_jpeg_info(cls, srcfile, dct_dims, image_dims, num_components, samp_factor, jpeg_color_space):
        status = cls.get().read_jpeg_info(cls.cstr(srcfile), dct_dims, image_dims, num_components, samp_factor, jpeg_color_space)
        if status == 0: raise IOError(f"reading of {srcfile} failed")
        
    @classmethod
    def read_jpeg_dct(cls, srcfile, dct, qt):
        status = cls.get().read_jpeg_dct(cls.cstr(srcfile), dct, qt)
        if status == 0: raise IOError(f"reading of {srcfile} DCT failed")

    @classmethod
    def write_jpeg_dct(cls, srcfile, dstfile, dct, image_dims, in_color_space, in_components, samp_factor, qt, quality):
        status = cls.get().write_jpeg_dct(cls.cstr(srcfile), cls.cstr(dstfile), dct, image_dims, in_color_space, in_components, samp_factor, qt, quality)
        if status == 0: raise IOError(f"writing DCT to {dstfile} failed")
    @classmethod
    def print_jpeg_params(cls, srcfile):
        status = cls.get().print_jpeg_params(cls.cstr(srcfile))
        if status == 0: raise IOError(f"reading of {srcfile} failed")

    @classmethod
    def read_jpeg_spatial(cls, srcfile, rgb, colormap, in_colormap, out_color_space, dither_mode, dct_method, flags):
        status = cls.get().read_jpeg_spatial(cls.cstr(srcfile), rgb, colormap, in_colormap,
                                             out_color_space, dither_mode, dct_method, cls.flags_to_mask(flags))
        if status == 0: raise IOError(f"reading of {srcfile} spatial failed")
    
    @classmethod
    def write_jpeg_spatial(cls, srcfile, dstfile, rgb, image_dims, in_color_space, in_components, dct_method, samp_factor, qt, quality, smoothing_factor, flags):
        status = cls.get().write_jpeg_spatial(cls.cstr(srcfile), cls.cstr(dstfile), rgb, image_dims, in_color_space, in_components,
                                              dct_method, samp_factor, qt, quality, smoothing_factor, cls.flags_to_mask(flags))
        if status == 0: raise IOError(f"writing RGB to {dstfile} failed")
        
    MASKS = {
        "DO_FANCY_UPSAMPLING": 0x1,
        "DO_BLOCK_SMOOTHING": 0x2,
        "TWO_PASS_QUANTIZE": 0x4,
        "ENABLE_1PASS_QUANT": 0x8,
        "ENABLE_EXTERNAL_QUANT": 0x10,
        "ENABLE_2PASS_QUANT": 0x20,
        "OPTIMIZE_CODING": 0x40,
        "PROGRESSIVE_MODE": 0x80,
        "QUANTIZE_COLORS": 0x100,
        "ARITH_CODE": 0x200,
        "WRITE_JFIF_HEADER": 0x400,
        "WRITE_ADOBE_MARKER": 0x800,
        "CCIR601_SAMPLING": 0x1000
    }

    @classmethod
    def flags_to_mask(cls, flags):
        mask = 0
        if flags is None: return mask
        for flag in flags:
            mask = mask | cls.MASKS[flag.upper()]
        return ctypes.c_ulong(mask)

    _lib = None
    version = None
    @classmethod
    def get(cls):
        # connect to library
        if cls._lib is None:
            cls._lib = cls._bind_lib()
        # return library
        return cls._lib
    
    @classmethod
    def set_version(cls, version):
        cls._lib = cls._bind_lib(version=version)
    @classmethod
    def get_version(cls):
        return cls.version
    @classmethod
    def _versions(cls):
        so_files = [f for f in os.listdir(cjpeglib.__path__[0]) if re.fullmatch(f'cjpeglib_.*\..*\.so', f)]
        return so_files
    @classmethod
    def versions(cls):
        vs = [re.search(f'cjpeg_[^.]*\..*\.so', f) for f in cls._versions()]
        vs = [v[0] for v in vs if v]
        return vs
    @classmethod
    def _bind_lib(cls, version='6b'):
        # path of the library
        so_files = [f for f in cls._versions() if re.fullmatch(f'cjpeglib_{version}\..*\.so', f)]
        try:
            so_file = so_files[0]
        except:
            raise Exception(f"dynamic library not found")
        libname = pathlib.Path(cjpeglib.__path__[0]) / so_file
        # connect
        cjpeglib_dylib = ctypes.CDLL(libname)
        cls.version = version
        return cjpeglib_dylib
    
    @staticmethod
    def cstr(s):
        if s is None: return None
        return ctypes.c_char_p(s.encode('utf-8'))
        