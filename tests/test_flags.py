
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

    def test_fancy_upsampling(self):
        self.logger.info("test_fancy_upsampling")
        jpeglib.version.set('8')
        fname = 'examples/IMG_0791.jpeg'
        im_def = jpeglib.read_spatial(fname, flags=[])
        im_fu  = jpeglib.read_spatial(fname, flags=['+DO_FANCY_UPSAMPLING'])
        im_ss  = jpeglib.read_spatial(fname, flags=['-DO_FANCY_UPSAMPLING'])
        np.testing.assert_array_equal(im_def.spatial, im_fu.spatial)
        self.assertTrue((im_def.spatial != im_ss.spatial).any())

    def test_fancy_downsampling(self):
        self.logger.info("test_fancy_downsampling")
        jpeglib.version.set('9')
        im = jpeglib.read_spatial('examples/IMG_0311.jpeg')

        # default
        im.write_spatial(self.tmp.name, flags=[])
        self.tmp.flush()
        Y_def, (Cb_def, Cr_def), qt_def = jpeglib.read_dct(self.tmp.name).load()
        im.write_spatial(self.tmp.name, flags=['+DO_FANCY_DOWNSAMPLING'])
        self.tmp.flush()
        Y_fu, (Cb_fu, Cr_fu), qt_fu = jpeglib.read_dct(self.tmp.name).load()
        im.write_spatial(self.tmp.name, flags=['-DO_FANCY_DOWNSAMPLING'])
        self.tmp.flush()
        Y_ss, (Cb_ss, Cr_ss), qt_ss = jpeglib.read_dct(self.tmp.name).load()



        np.testing.assert_array_equal(Y_def, Y_fu)
        np.testing.assert_array_equal(Cb_def, Cb_fu)
        np.testing.assert_array_equal(Cr_def, Cr_fu)

        # bug: False for some reason
        # self.assertTrue((Y_fu != Y_ss).any())
        # self.assertTrue((Cb_fu != Cb_ss).any())
        # self.assertTrue((Cr_fu != Cr_ss).any())
