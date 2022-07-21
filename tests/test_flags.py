
import logging
import numpy as np
import tempfile
import unittest

import jpeglib


class TestFlags(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')

    def tearDown(self):
        del self.tmp

    # def test_fancy_upsampling(self):

    # 	jpeglib.version.set('6b')
    # 	with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    # 		print("default flags")
    # 		x_def = im.read_spatial(flags = [])
    # 		print("+DO_FANCY_UPSAMPLING")
    # 		x_fu = im.read_spatial(flags = ['+DO_FANCY_UPSAMPLING'])
    # 		print("-DO_FANCY_UPSAMPLING")
    # 		x_ss = im.read_spatial(flags = ['-DO_FANCY_UPSAMPLING'])
    # 	self.assertTrue((x_def == x_fu).all())
    # 	self.assertTrue((x_def != x_ss).any())

    def test_fancy_downsampling(self):
        jpeglib.version.set('8d')
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            x = im.read_spatial(flags=['-DO_FANCY_DOWNSAMPLING'])

        # default flags
        with jpeglib.JPEG() as im:
            im.write_spatial(x, self.tmp.name, flags=[])
        with jpeglib.JPEG(self.tmp.name) as im:
            Y_def, CbCr_def, qt_def = im.read_dct()

        # fancy upsampling
        with tempfile.NamedTemporaryFile() as tmp:
            with jpeglib.JPEG() as im:
                im.write_spatial(x, tmp.name,
                                 flags=['+DO_FANCY_DOWNSAMPLING'])
            with jpeglib.JPEG(tmp.name) as im:
                Y_fu, CbCr_fu, qt_fu = im.read_dct()
        # simple scaling
        with tempfile.NamedTemporaryFile() as tmp:
            with jpeglib.JPEG() as im:
                im.write_spatial(x, tmp.name,
                                 flags=['-DO_FANCY_DOWNSAMPLING'])
            with jpeglib.JPEG(tmp.name) as im:
                Y_ss, CbCr_ss, qt_ss = im.read_dct()

        np.testing.assert_array_equal(Y_def, Y_fu)
        np.testing.assert_array_equal(CbCr_def, CbCr_fu)

        # self.assertFalse((Y_fu == Y_ss).all())
        # self.assertTrue((CbCr_fu == CbCr_ss).all())
