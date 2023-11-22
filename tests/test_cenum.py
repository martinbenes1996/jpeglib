"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import logging
import os
from parameterized import parameterized
import tempfile
import unittest

import jpeglib


class TestCEnum(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    @parameterized.expand([
        ['JDCT_ISLOW', jpeglib.DCTMethod.JDCT_ISLOW, 0],
        ['JDCT_IFAST', jpeglib.DCTMethod.JDCT_IFAST, 1],
        ['JDCT_FLOAT', jpeglib.DCTMethod.JDCT_FLOAT, 2],
    ])
    def test_dct_method(self, dct_name, dct_ref, i):
        """Test class DCTMethod."""
        self.logger.info(f'test_dct_method_{dct_name}_{i}')
        #
        dct = jpeglib.DCTMethod[dct_name]
        self.assertEqual(dct, dct_ref)
        self.assertEqual(str(dct), dct_name)
        self.assertEqual(int(dct), i)

    def test_dct_method_diff(self):
        """Test difference between DCT methods."""
        self.logger.info('test_dct_method_diff')
        #
        im = {
            k: jpeglib.read_spatial(
                'tests/assets/IMG_0311.jpeg',
                dct_method=jpeglib.DCTMethod[k],
            )
            for k in ['JDCT_ISLOW', 'JDCT_IFAST', 'JDCT_FLOAT']
        }
        self.assertFalse((im['JDCT_ISLOW'].spatial == im['JDCT_IFAST'].spatial).all())
        self.assertFalse((im['JDCT_ISLOW'].spatial == im['JDCT_FLOAT'].spatial).all())
        self.assertFalse((im['JDCT_IFAST'].spatial == im['JDCT_FLOAT'].spatial).all())

    def test_dct_method_invalid(self):
        """Test invalid DCT method."""
        self.logger.info('test_dct_method_invalid')
        self.assertRaises(
            ValueError,
            jpeglib.DCTMethod,
            'JDCT_SUPERFAST'
        )

    @parameterized.expand([
        ['JCS_UNKNOWN', jpeglib.Colorspace.JCS_UNKNOWN, None, 0],
        ['JCS_GRAYSCALE', jpeglib.Colorspace.JCS_UNKNOWN, 1, 1],
        ['JCS_RGB', jpeglib.Colorspace.JCS_UNKNOWN, 3, 2],
        ['JCS_YCbCr', jpeglib.Colorspace.JCS_UNKNOWN, 3, 3],
        ['JCS_CMYK', jpeglib.Colorspace.JCS_UNKNOWN, 4, 4],
        ['JCS_YCCK', jpeglib.Colorspace.JCS_UNKNOWN, 4, 5],
    ])
    def test_color_space(self, cspace_name, cspace_ref, channels, i):
        """Test class Colorspace."""
        self.logger.info(f'test_color_space_{cspace_name}_{channels}_{i}')
        cspace = jpeglib.Colorspace[cspace_name]
        self.assertEqual(cspace, cspace_ref)
        self.assertEqual(str(cspace), cspace_name)
        self.assertEqual(int(cspace), i)
        if cspace_name == 'JCS_UNKNOWN':
            self.assertRaises(Exception, lambda: cspace.channels)
        else:
            self.assertEqual(cspace.channels, channels)

    def test_color_space_invalid(self):
        """Test invalid color space."""
        self.logger.info('test_color_space_invalid')
        self.assertRaises(
            ValueError,
            jpeglib.DCTMethod,
            'JCS_HSV'
        )

    @parameterized.expand([
        ['JPEG_RST0', jpeglib.MarkerType.JPEG_RST0, 0xD0],
        ['JPEG_RST1', jpeglib.MarkerType.JPEG_RST1, 0xD1],
        ['JPEG_RST2', jpeglib.MarkerType.JPEG_RST2, 0xD2],
        ['JPEG_RST3', jpeglib.MarkerType.JPEG_RST3, 0xD3],
        ['JPEG_RST4', jpeglib.MarkerType.JPEG_RST4, 0xD4],
        ['JPEG_RST5', jpeglib.MarkerType.JPEG_RST5, 0xD5],
        ['JPEG_RST6', jpeglib.MarkerType.JPEG_RST6, 0xD6],
        ['JPEG_RST7', jpeglib.MarkerType.JPEG_RST7, 0xD7],
        ['JPEG_RST8', jpeglib.MarkerType.JPEG_RST8, 0xD8],
        ['JPEG_EOI', jpeglib.MarkerType.JPEG_EOI, 0xD9],
        ['JPEG_APP0', jpeglib.MarkerType.JPEG_APP0, 0xE0],
        ['JPEG_APP1', jpeglib.MarkerType.JPEG_APP1, 0xE1],
        ['JPEG_APP2', jpeglib.MarkerType.JPEG_APP2, 0xE2],
        ['JPEG_APP3', jpeglib.MarkerType.JPEG_APP3, 0xE3],
        ['JPEG_APP4', jpeglib.MarkerType.JPEG_APP4, 0xE4],
        ['JPEG_APP5', jpeglib.MarkerType.JPEG_APP5, 0xE5],
        ['JPEG_APP6', jpeglib.MarkerType.JPEG_APP6, 0xE6],
        ['JPEG_APP7', jpeglib.MarkerType.JPEG_APP7, 0xE7],
        ['JPEG_APP8', jpeglib.MarkerType.JPEG_APP8, 0xE8],
        ['JPEG_APP9', jpeglib.MarkerType.JPEG_APP9, 0xE9],
        ['JPEG_APP10', jpeglib.MarkerType.JPEG_APP10, 0xEA],
        ['JPEG_APP11', jpeglib.MarkerType.JPEG_APP11, 0xEB],
        ['JPEG_APP12', jpeglib.MarkerType.JPEG_APP12, 0xEC],
        ['JPEG_APP13', jpeglib.MarkerType.JPEG_APP13, 0xED],
        ['JPEG_APP14', jpeglib.MarkerType.JPEG_APP14, 0xEE],
        ['JPEG_APP15', jpeglib.MarkerType.JPEG_APP15, 0xEF],
        ['JPEG_COM', jpeglib.MarkerType.JPEG_COM, 0xFE],
    ])
    def test_marker_type(self, mtype_name, mtype_ref, i):
        """"""
        self.logger.info(f'test_marker_type({mtype_name})')
        mtype = jpeglib.MarkerType[mtype_name]
        self.assertEqual(mtype, mtype_ref)
        self.assertEqual(str(mtype), mtype_name)
        self.assertEqual(int(mtype), i)

    def test_marker_type_invalid(self):
        """"""
        self.logger.info('test_marker_type_invalid')
        self.assertRaises(
            ValueError,
            jpeglib.MarkerType,
            'JPEG_SOI',
        )

    def test_abbr_cenum(self):
        """Test abbreviated CEnum."""
        self.logger.info('test_abbr_cenum')
        self.assertEqual(jpeglib.Colorspace.JCS_UNKNOWN, jpeglib.JCS_UNKNOWN)
        self.assertEqual(jpeglib.Colorspace.JCS_GRAYSCALE, jpeglib.JCS_GRAYSCALE)
        self.assertEqual(jpeglib.Colorspace.JCS_RGB, jpeglib.JCS_RGB)
        self.assertEqual(jpeglib.Colorspace.JCS_YCbCr, jpeglib.JCS_YCbCr)
        self.assertEqual(jpeglib.Colorspace.JCS_CMYK, jpeglib.JCS_CMYK)
        self.assertEqual(jpeglib.Colorspace.JCS_YCCK, jpeglib.JCS_YCCK)
        self.assertEqual(jpeglib.DCTMethod.JDCT_ISLOW, jpeglib.JDCT_ISLOW)
        self.assertEqual(jpeglib.DCTMethod.JDCT_IFAST, jpeglib.JDCT_IFAST)
        self.assertEqual(jpeglib.DCTMethod.JDCT_FLOAT, jpeglib.JDCT_FLOAT)
        self.assertEqual(jpeglib.MarkerType.JPEG_APP0, jpeglib.JPEG_APP0)
        self.assertEqual(jpeglib.MarkerType.JPEG_APP1, jpeglib.JPEG_APP1)
        self.assertEqual(jpeglib.MarkerType.JPEG_COM, jpeglib.JPEG_COM)


__all__ = ["TestCEnum"]
