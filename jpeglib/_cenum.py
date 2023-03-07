"""Module with C-enumerations.

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

from __future__ import annotations
from dataclasses import dataclass
import enum


class _Enum(enum.Enum):
    def __int__(self):
        return self.value
    def __str__(self):
        return self.name


@dataclass
class Colorspace(_Enum):
    """Representation of color space.

    Allows mapping name-index and acquiral of number of channels.
    """
    JCS_UNKNOWN   = 0  # Unspecified color space
    JCS_GRAYSCALE = 1  # Monochrome
    JCS_RGB       = 2  # Standard RGB
    JCS_YCbCr     = 3  # YCbCr or YUB, standard YCC
    JCS_CMYK      = 4  # CMYK
    JCS_YCCK      = 5  # YCbCrK

    @property
    def channels(self) -> int:
        channel_no = {
            "JCS_GRAYSCALE": 1,
            "JCS_RGB":       3,
            "JCS_YCbCr":     3,
            "JCS_CMYK":      4,
            "JCS_YCCK":      4,
        }
        return channel_no[self.name]

class DCTMethod(_Enum):
    JDCT_ISLOW = 0
    JDCT_IFAST = 1
    JDCT_FLOAT = 2


class Dithermode(_Enum):
    JDITHER_NONE    = 0
    JDITHER_ORDERED = 1
    JDITHER_FS      = 2


class MarkerType(_Enum):
    JPEG_RST0  = 0xD0
    JPEG_RST1  = 0xD0+1
    JPEG_RST2  = 0xD0+2
    JPEG_RST3  = 0xD0+3
    JPEG_RST4  = 0xD0+4
    JPEG_RST5  = 0xD0+5
    JPEG_RST6  = 0xD0+6
    JPEG_RST7  = 0xD0+7
    JPEG_RST8  = 0xD0+8
    JPEG_EOI   = 0xD9
    JPEG_APP0  = 0xE0
    JPEG_APP1  = 0xE0+1
    JPEG_APP2  = 0xE0+2
    JPEG_APP3  = 0xE0+3
    JPEG_APP4  = 0xE0+4
    JPEG_APP5  = 0xE0+5
    JPEG_APP6  = 0xE0+6
    JPEG_APP7  = 0xE0+7
    JPEG_APP8  = 0xE0+8
    JPEG_APP9  = 0xE0+9
    JPEG_APP10 = 0xE0+10
    JPEG_APP11 = 0xE0+11
    JPEG_APP12 = 0xE0+12
    JPEG_APP13 = 0xE0+13
    JPEG_APP14 = 0xE0+14
    JPEG_APP15 = 0xE0+15
    JPEG_COM   = 0xFE

