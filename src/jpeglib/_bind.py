"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import ctypes
import os
import pathlib
import re
from typing import List

from . import cjpeglib


class CJpegLib:
    """"""

    @classmethod
    def jpeg_lib_version(cls):
        return cls.get().jpeg_lib_version()

    @classmethod
    def read_jpeg_info(
        cls,
        srcfile: str,
        block_dims,
        image_dims,
        num_components,
        samp_factor,
        jpeg_color_space: int,
        marker_lengths,
        marker_types,
        huffman_bits,
        huffman_values,
        num_scans,
        flags: List[str],
    ):
        status = cls.get().read_jpeg_info(
            cls.cstr(srcfile),
            block_dims,
            image_dims,
            num_components,
            samp_factor,
            jpeg_color_space,
            marker_lengths,
            marker_types,
            huffman_bits,
            huffman_values,
            num_scans,
            flags
        )
        if status == 0:
            raise IOError(f"reading info of {srcfile} failed")

    @classmethod
    def read_jpeg_markers(cls, srcfile: str, markers):
        status = cls.get().read_jpeg_markers(cls.cstr(srcfile), markers)
        if status == 0:
            raise IOError(f"reading markers of {srcfile} failed")

    @classmethod
    def read_jpeg_dct(
        cls,
        srcfile: str,
        Y,
        Cb,
        Cr,
        K,
        qt,
        quant_tbl_no,
        path=None,
    ):
        if path is None:
            path = srcfile
        status = cls.get().read_jpeg_dct(
            cls.cstr(srcfile),
            Y,
            Cb,
            Cr,
            K,
            qt,
            quant_tbl_no
        )
        if status == 0:
            print(f"{path} {srcfile}")
            raise IOError(f"reading of {path} DCT failed")

    @classmethod
    def write_jpeg_dct(
        cls,
        srcfile: str,
        dstfile: str,
        Y,
        Cb,
        Cr,
        K,
        image_dims,
        block_dims,
        in_color_space,
        in_components,
        samp_factor,
        qt,
        quality,
        quant_tbl_no,
        num_markers,
        marker_types,
        marker_lengths,
        markers,
        flags: List[str],
    ):
        status = cls.get().write_jpeg_dct(
            cls.cstr(srcfile),
            cls.cstr(dstfile),
            Y,
            Cb,
            Cr,
            K,
            image_dims,
            block_dims,
            samp_factor,
            in_color_space,
            in_components,
            qt,
            quality,
            quant_tbl_no,
            num_markers,
            marker_types,
            marker_lengths,
            markers,
            cls.flags_to_mask(flags),
        )
        if status == 0:
            raise IOError(f"writing DCT to {dstfile} failed")

    @classmethod
    def print_jpeg_params(cls, srcfile: str):
        status = cls.get().print_jpeg_params(cls.cstr(srcfile))
        if status == 0:
            raise IOError(f"reading of {srcfile} failed")

    @classmethod
    def read_jpeg_spatial(
        cls,
        srcfile: str,
        spatial,
        colormap,
        in_colormap,
        out_color_space,
        dither_mode,
        dct_method,
        flags,
        path=None,
    ):
        if path is None:
            path = srcfile
        status = cls.get().read_jpeg_spatial(
            cls.cstr(srcfile),
            spatial,
            colormap,
            in_colormap,
            out_color_space,
            dither_mode,
            dct_method,
            cls.flags_to_mask(flags)
        )
        if status == 0:
            raise IOError(f"reading of {path} spatial failed")

    @classmethod
    def write_jpeg_spatial(
        cls,
        dstfile: str,
        spatial,
        image_dims,
        jpeg_color_space,
        num_components,
        dct_method,
        samp_factor,
        qt,
        quality,
        quant_tbl_no,
        base_quant_tbl_idx,
        smoothing_factor,
        num_markers: int,
        marker_types,
        marker_lengths,
        markers,
        num_scans,
        scan_script,
        huffman_bits,
        huffman_values,
        progressive_mode,
        flags: List[str],
    ):
        status = cls.get().write_jpeg_spatial(
            cls.cstr(dstfile),
            spatial,
            image_dims,
            jpeg_color_space,
            num_components,
            dct_method,
            samp_factor,
            qt,
            cls.factor(quality),
            quant_tbl_no,
            cls.factor(base_quant_tbl_idx),
            cls.factor(smoothing_factor),
            num_markers,
            marker_types,
            marker_lengths,
            markers,
            num_scans,
            scan_script,
            huffman_bits,
            huffman_values,
            cls.flags_to_mask(flags, progressive_mode),
        )
        if status == 0:
            raise IOError(f"writing spatial to {dstfile} failed")

    @classmethod
    def read_jpeg_progressive(
        cls,
        srcfile: str,
        spatial,
        colormap,
        in_colormap,
        out_color_space,
        dither_mode,
        dct_method,
        scan_script,
        huffman_bits,
        huffman_values,
        qt,
        quant_tbl_no,
        flags,
        path=None,
    ):
        status = cls.get().read_jpeg_progressive(
            cls.cstr(srcfile),
            spatial,
            colormap,
            in_colormap,
            out_color_space,
            dither_mode,
            dct_method,
            scan_script,
            huffman_bits,
            huffman_values,
            qt,
            quant_tbl_no,
            cls.flags_to_mask(flags, True),
        )
        if status == 0:
            raise IOError(f"reading scanscript of {srcfile} failed")

    MASKS = {
        "DO_FANCY_SAMPLING": (0b1 << 0),
        "DO_FANCY_UPSAMPLING": (0b1 << 0),
        "DO_FANCY_DOWNSAMPLING": (0b1 << 0),
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
        "TRELLIS_QUANT": (0b1 << 28),
        "TRELLIS_QUANT_DC": (0b1 << 30),
        "TRELLIS_Q_OPT": (0b1 << 32),
        "OPTIMIZE_SCANS": (0b1 << 34),
        "USE_SCANS_IN_TRELLIS": (0b1 << 36),
        "OVERSHOOT_DERINGING": (0b1 << 38),
    }

    @classmethod
    def flags_to_mask(
        cls,
        flags: List[str],
        progressive_mode: bool = None,
    ):
        # Create a 64-bit mask with all 1s
        mask = 0xFFFFFFFFFFFFFFFF
        
        if flags is None:
            return mask
        
        # progressive mode
        if progressive_mode is not None:
            mask ^= cls.MASKS['PROGRESSIVE_MODE'] << 1
            if not progressive_mode:
                mask ^= cls.MASKS['PROGRESSIVE_MODE']
        
        # manual flags
        for flag in flags:
            # parse sign
            sign = '-' if flag[0] == '-' else '+'
            if not flag[0].isalpha():
                # flag does not have a sign, so we interpret it as +
                flag = flag[1:]
            
            # The more significant bit indicates whether to keep the default (defbit = 1) or change (defbit = 0).
            defbit = cls.MASKS[flag.upper()] << 1
            
            # The less significant bit contains the changed value.
            flagbit = cls.MASKS[flag.upper()]
            
            # Set defbit to 0, meaning that the default value should be overwritten.
            mask ^= defbit
            
            # Set the new value. By default, it is set to 1.
            if sign == '-':
                # Set new value to 0
                mask ^= flagbit

        return ctypes.c_ulonglong(mask)

    @classmethod
    def mask_to_flags(cls, mask: int):
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
            factor = -1
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
        """
        Find all library files in cjpeglib directory.
        Library files must start with "cjpeglib" and the file extensions must be one of the following:
        - ".so" (Linux and Mac),
        - ".dll" (Windows Cygwin), or
        - ".pyd" (similar to a Windows DLL).
        Return a list of matching filenames.
        """
        so_files = [
            f
            for f in os.listdir(list(cjpeglib.__path__)[0])
            if re.fullmatch(r'cjpeglib_.*\.(so|dll|pyd)', f)
        ]
        return so_files

    @classmethod
    def versions(cls):
        """
        Find all library files in the cjpeglib directory and extract the version names
        Return a list of the version names.
        """
        vs = [re.search(r'(?<=cjpeglib_)[^.]*', f)[0] for f in cls._versions()]
        return vs

    @classmethod
    def _bind_lib(cls, version='6b'):
        # List all library files
        all_so_files = cls._versions()

        # Select the desired version
        matching_so_files = list(filter(lambda f: re.fullmatch(rf'cjpeglib_{version}\..*\.(so|dll|pyd)', f), all_so_files))
        if len(matching_so_files) == 0:
            raise RuntimeError(f'version "{version}" not found')
        elif len(matching_so_files) > 1:
            raise RuntimeError(f'found several versions matching the given version string "{version}"')

        # Found exactly one library file
        so_file = matching_so_files[0]

        # Obtain full path
        lib_full_path = str(pathlib.Path(list(cjpeglib.__path__)[0]) / so_file)

        # Connect
        cjpeglib_dylib = ctypes.CDLL(lib_full_path)
        cls.version = version
        return cjpeglib_dylib

    @staticmethod
    def cstr(s):
        if s is None:
            return None
        return ctypes.c_char_p(s.encode('utf-8'))
