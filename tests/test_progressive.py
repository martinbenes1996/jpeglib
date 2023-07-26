"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import tempfile
import unittest

from _defs import LIBJPEG_VERSIONS
import jpeglib


class TestProgressive(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    def test_read_progressive_flag(self):
        self.logger.info("test_read_progressive_flag")

        im = jpeglib.read_spatial("examples/images-6b/testprog.jpg")
        self.assertTrue(im.progressive_mode)

        im = jpeglib.read_spatial("examples/images-6b/testimg.jpg")
        self.assertFalse(im.progressive_mode)

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_progressive_decompress_vs_pil(self, version):
        """Test on test images from libjpeg."""
        self.logger.info(f"test_progressive_decompress_vs_pil_{version}")
        jpeglib.version.set(version)

        # im_prog = jpeglib.read_spatial(
        _ = jpeglib.read_spatial(
            f'examples/images-{version}/testprog.jpg',
            flags=[
                '+PROGRESSIVE_MODE',
                '+DO_FANCY_UPSAMPLING',
                '+DO_BLOCK_SMOOTHING'
            ]
        )
        # im_prog = jpeglib.read_spatial(
        _ = np.array(Image.open(
            f'examples/images-{version}/testprog.jpg'
        ))
        # np.testing.assert_array_almost_equal(im_prog.spatial, rgb_pil) # TODO: Nora

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_progressive_dct(self, version):
        self.logger.info(f"test_progressive_dct_{version}")
        # load dct - to fix
        im = jpeglib.read_dct(f'examples/images-{version}/testprog.jpg')
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(im.Y, im2.Y)
        # D = np.abs((CbCr.astype(np.int) - CbCr2.astype(np.int)))
        # print((D != 0).sum()) # 1073 mismatches
        # np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(im.qt, im2.qt)

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_progressive_sequential(self, version):
        self.logger.info(f"test_progressive_sequential_{version}")

        # load progressive image
        im_seq = jpeglib.read_spatial(
            f'examples/images-{version}/testimg.jpg',
            flags=['-PROGRESSIVE_MODE']
        )
        im_prog = jpeglib.read_spatial(
            f'examples/images-{version}/testimgp.jpg',
            flags=['+PROGRESSIVE_MODE']
        )
        np.testing.assert_array_almost_equal(im_seq.spatial, im_prog.spatial)

    def test_progressive_standard_scanscript(self):
        self.logger.info('test_progressive_standard_scanscript')

        # TODO: libjpeg produces the standard scanscript during compression

    def test_progressive_same_scanscript(self):
        self.logger.info('test_progressive_same_scanscript')

        # TODO:

    def test_progressive_set_scanscript(self):
        self.logger.info('test_progressive_set_scanscript')

        # TODO: create custom scanscript and compress with it

    # TODO: tests for social network scan scripts?
    # TODO: ?

__all__ = ["TestProgressive"]
