
import logging
import os
from parameterized import parameterized
import tempfile
import unittest

import jpeglib


class TestCStruct(unittest.TestCase):
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
        [dct_name, i]
        for i, dct_name in enumerate([
            'JDCT_ISLOW', 'JDCT_IFAST', 'JDCT_FLOAT'
        ])
    ])
    def test_dct_method(self, dct_name, i):
        """Test class DCTMethod."""
        self.logger.info(f'test_dct_method_{dct_name}_{i}')
        dct = jpeglib.DCTMethod(name=dct_name)
        self.assertIsInstance(dct.name, str)
        self.assertEqual(dct.name, dct_name)
        self.assertIsInstance(dct.index, int)
        self.assertEqual(dct.index, i)

    def test_dct_method_invalid(self):
        """Test invalid DCT method."""
        self.logger.info('test_dct_method_invalid')
        self.assertRaises(
            KeyError,
            jpeglib.DCTMethod,
            'JDCT_SUPERFAST'
        )

    @parameterized.expand([
        [dct_name, channels, i]
        for i, (dct_name, channels) in enumerate(({
            'JCS_UNKNOWN': None,
            'JCS_GRAYSCALE': 1,
            'JCS_RGB': 3,
            'JCS_YCbCr': 3,
            'JCS_CMYK': 4,
            'JCS_YCCK': 4,
        }).items())
    ])
    def test_color_space(self, cspace_name, channels, i):
        """Test class Colorspace."""
        self.logger.info(f'test_color_space_{cspace_name}_{channels}_{i}')
        cspace = jpeglib.Colorspace(name=cspace_name)
        self.assertEqual(cspace, jpeglib.Colorspace(name=cspace_name))
        self.assertIsInstance(cspace.name, str)
        self.assertEqual(cspace.name, cspace_name)
        self.assertIsInstance(cspace.index, int)
        self.assertEqual(cspace.index, i)
        if cspace_name == 'JCS_UNKNOWN':
            self.assertRaises(Exception, lambda : cspace.channels)
        else:
            self.assertEqual(cspace.channels, channels)

    def test_color_space_invalid(self):
        """Test invalid color space."""
        self.logger.info('test_color_space_invalid')
        self.assertRaises(
            KeyError,
            jpeglib.DCTMethod,
            'JCS_HSV'
        )


__all__ = ["TestCStruct"]
