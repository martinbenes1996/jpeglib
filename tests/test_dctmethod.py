
import logging
import tempfile
import unittest

import jpeglib


class TestDCTMethod(unittest.TestCase):
    logger = logging.getLogger()

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='jpeg')

    def tearDown(self):
        del self.tmp

    def test_islow(self):
        """DCT method islow."""
        self.logger.info("test_islow")
        dct = jpeglib.DCTMethod(name='JDCT_ISLOW')
        self.assertIsInstance(dct.name, str)
        self.assertEqual(dct.name, 'JDCT_ISLOW')
        self.assertIsInstance(dct.index, int)
        self.assertEqual(dct.index, 0)

    def test_ifast(self):
        """DCT method ifast."""
        self.logger.info("test_ifast")
        dct = jpeglib.DCTMethod(name='JDCT_IFAST')
        self.assertIsInstance(dct.name, str)
        self.assertEqual(dct.name, 'JDCT_IFAST')
        self.assertIsInstance(dct.index, int)
        self.assertEqual(dct.index, 1)

    def test_float(self):
        """DCT method float."""
        self.logger.info("test_float")
        dct = jpeglib.DCTMethod(name='JDCT_FLOAT')
        self.assertIsInstance(dct.name, str)
        self.assertEqual(dct.name, 'JDCT_FLOAT')
        self.assertIsInstance(dct.index, int)
        self.assertEqual(dct.index, 2)


__all__ = ["TestDCTMethod"]
