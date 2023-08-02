"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
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



__all__ = ["TestCEnum"]
