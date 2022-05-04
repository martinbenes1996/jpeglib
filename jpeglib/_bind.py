
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
    def read_jpeg_info(cls,
                       srcfile: str, block_dims, image_dims, num_components,
                       samp_factor, jpeg_color_space, marker_lengths, marker_types, flags):
        status = cls.get().read_jpeg_info(cls.cstr(srcfile), block_dims, image_dims, num_components,
                                          samp_factor, jpeg_color_space, marker_lengths, marker_types, flags)
        if status == 0:
            raise IOError(f"reading info of {srcfile} failed")

    @classmethod
    def read_jpeg_markers(cls, srcfile: str, markers):
        status = cls.get().read_jpeg_markers(cls.cstr(srcfile), markers)
        if status == 0:
            raise IOError(f"reading markers of {srcfile} failed")

    @classmethod
    def read_jpeg_dct(cls, srcfile: str, Y, Cb, Cr, qt, quant_tbl_no, path=None):
        if path is None: path = srcfile
        status = cls.get().read_jpeg_dct(cls.cstr(srcfile), Y, Cb, Cr, qt, quant_tbl_no)
        if status == 0:
            raise IOError(f"reading of {path} DCT failed")

    @classmethod
    def write_jpeg_dct(cls,
                       srcfile: str, dstfile: str, Y, Cb, Cr,
                       image_dims, block_dims, in_color_space, in_components,
                       qt, quality, num_markers, marker_types, marker_lengths, markers):
        status = cls.get().write_jpeg_dct(cls.cstr(srcfile), cls.cstr(dstfile), Y, Cb, Cr,
                                          image_dims, block_dims, in_color_space, in_components,
                                          qt, quality, num_markers, marker_types, marker_lengths, markers)
        if status == 0:
            raise IOError(f"writing DCT to {dstfile} failed")

    @classmethod
    def print_jpeg_params(cls, srcfile: str):
        status = cls.get().print_jpeg_params(cls.cstr(srcfile))
        if status == 0:
            raise IOError(f"reading of {srcfile} failed")

    @classmethod
    def read_jpeg_spatial(cls,
                          srcfile: str, spatial, colormap, in_colormap, out_color_space,
                          dither_mode, dct_method, flags,
                          path = None):
        if path is None: path = srcfile
        status = cls.get().read_jpeg_spatial(cls.cstr(srcfile), spatial, colormap, in_colormap,
                                             out_color_space, dither_mode, dct_method, cls.flags_to_mask(flags))
        if status == 0:
            raise IOError(f"reading of {path} spatial failed")

    @classmethod
    def write_jpeg_spatial(cls,
                           dstfile: str, spatial, image_dims, jpeg_color_space, num_components,
                           dct_method, samp_factor, qt, quality, smoothing_factor,
                           num_markers: int, marker_types, marker_lengths, markers,
                           flags
                           ):
        status = cls.get().write_jpeg_spatial(cls.cstr(dstfile), spatial, image_dims, jpeg_color_space, num_components,
                                              dct_method, samp_factor, qt, cls.factor(
                                                  quality), cls.factor(smoothing_factor),
                                              num_markers, marker_types, marker_lengths, markers,
                                              cls.flags_to_mask(flags))
        if status == 0:
            raise IOError(f"writing RGB to {dstfile} failed")

    MASKS = {
        "DO_FANCY_SAMPLING": (0b1 << 0), "DO_FANCY_UPSAMPLING": (0b1 << 0), "DO_FANCY_DOWNSAMPLING": (0b1 << 0),
        "DO_BLOCK_SMOOTHING": (0b1 << 2),
        "TWO_PASS_QUANTIZE": (0b1 << 4),
        "ENABLE_1PASS_QUANT": (0b1 << 6),
        "ENABLE_EXTERNAL_QUANT": (0b1 << 8),
        "ENABLE_2PASS_QUANT": (0b1 << 10),
        "OPTIMIZE_CODING": (0b1 << 12),
        "PROGRESSIVE_MODE": (0b1 << 14),
        "QUANTIZE_COLORS": (0b1 << 16),
        "ARITH_CODE": (0b1 << 18),
        "WRITE_JFIF_HEADER": (0b1 << 20),
        "WRITE_ADOBE_MARKER": (0b1 << 22),
        "CCIR601_SAMPLING": (0b1 << 24),
        "FORCE_BASELINE": (0b1 << 26),
    }

    @classmethod
    def flags_to_mask(cls, flags):
        mask = 0xFFFFFFFF
        if flags is None:
            return mask
        for flag in flags:
            # print(f"flag {flag}:")
            # parse sign
            sign = '-' if flag[0] == '-' else '+'
            if not flag[0].isalpha():
                flag = flag[1:]
            # get flags
            flagbit = cls.MASKS[flag.upper()]
            defbit = cls.MASKS[flag.upper()] << 1
            # map
            mask ^= defbit  # reset default value
            if sign == '-':
                mask ^= (flagbit)  # erase bit

        return ctypes.c_ulonglong(mask)

    @classmethod
    def mask_to_flags(cls, mask):
        flags = []
        bitmask = mask[0]
        # PROGRESSIVE_MODE = 0b00100
        # mask = ??1?? or ??0??
        if ((cls.MASKS["PROGRESSIVE_MODE"]) & bitmask) != 0:
            flags.append("PROGRESSIVE_MODE")

        return flags

    @classmethod
    def factor(cls, factor):
        if factor is None:
            factor = 0
        return ctypes.c_short(factor)

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
        so_files = [f for f in os.listdir(
            list(cjpeglib.__path__)[0]) if re.fullmatch(f'cjpeglib_.*\..*\.so', f)]
        return so_files

    @classmethod
    def versions(cls):
        vs = [re.search(f'cjpeglib_[^.]*\..*\.so', f) for f in cls._versions()]
        vs = [v[0] for v in vs if v]
        return vs

    @classmethod
    def _bind_lib(cls, version='6b'):
        # path of the library
        so_files = [f for f in cls._versions() if re.fullmatch(
            f'cjpeglib_{version}\..*\.so', f)]
        try:
            so_file = so_files[0]
        except:
            raise Exception(f"dynamic library not found")
        libname = pathlib.Path(list(cjpeglib.__path__)[0]) / so_file
        # connect
        cjpeglib_dylib = ctypes.CDLL(libname)
        cls.version = version
        return cjpeglib_dylib

    @staticmethod
    def cstr(s):
        if s is None:
            return None
        return ctypes.c_char_p(s.encode('utf-8'))
