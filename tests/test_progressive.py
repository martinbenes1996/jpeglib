
import logging
import numpy as np
from PIL import Image
import tempfile
import unittest

import jpeglib


class TestProgressive(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='jpeg')

    def tearDown(self):
        self.tmp.close()
        del self.tmp

    def test_read_progressive_flag(self):
        self.logger.info("test_read_progressive_flag")

        im = jpeglib.read_spatial("examples/images-6b/testprog.jpg")
        self.assertTrue(im.progressive_mode)

        im = jpeglib.read_spatial("examples/images-6b/testimg.jpg")
        self.assertFalse(im.progressive_mode)

    def _test_progressive_decompress_vs_pil(self, version):
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

    def _test_progressive_dct(self, version):
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

    def _test_progressive_sequential(self, version):
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

    def test_progressive_decompress_vs_pil_6b(self):
        self._test_progressive_decompress_vs_pil('6b')

    def test_progressive_decompress_vs_pil_7(self):
        self._test_progressive_decompress_vs_pil('7')

    def test_progressive_decompress_vs_pil_8(self):
        self._test_progressive_decompress_vs_pil('8')

    def test_progressive_decompress_vs_pil_8a(self):
        self._test_progressive_decompress_vs_pil('8a')

    def test_progressive_decompress_vs_pil_8b(self):
        self._test_progressive_decompress_vs_pil('8b')

    def test_progressive_decompress_vs_pil_8c(self):
        self._test_progressive_decompress_vs_pil('8c')

    def test_progressive_decompress_vs_pil_8d(self):
        self._test_progressive_decompress_vs_pil('8d')

    def test_progressive_decompress_vs_pil_9(self):
        self._test_progressive_decompress_vs_pil('9')

    def test_progressive_decompress_vs_pil_9a(self):
        self._test_progressive_decompress_vs_pil('9a')

    def test_progressive_decompress_vs_pil_9b(self):
        self._test_progressive_decompress_vs_pil('9b')

    def test_progressive_decompress_vs_pil_9c(self):
        self._test_progressive_decompress_vs_pil('9c')

    def test_progressive_decompress_vs_pil_9d(self):
        self._test_progressive_decompress_vs_pil('9d')

    def test_progressive_decompress_vs_pil_9e(self):
        self._test_progressive_decompress_vs_pil('9e')

    # def test_progressive_decompress_vs_pil_turbo210(self):
    #     self._test_progressive_decompress_vs_pil('turbo210')

    # def test_progressive_decompress_vs_pil_mozjpeg300(self):
    #     self._test_progressive_decompress_vs_pil('mozjpeg300')

    def test_progressive_dct_6b(self):
        self._test_progressive_dct('6b')

    def test_progressive_dct_7(self):
        self._test_progressive_dct('7')

    def test_progressive_dct_8(self):
        self._test_progressive_dct('8')

    def test_progressive_dct_8a(self):
        self._test_progressive_dct('8a')

    def test_progressive_dct_8b(self):
        self._test_progressive_dct('8b')

    def test_progressive_dct_8c(self):
        self._test_progressive_dct('8c')

    def test_progressive_dct_8d(self):
        self._test_progressive_dct('8d')

    def test_progressive_dct_9(self):
        self._test_progressive_dct('9')

    def test_progressive_dct_9a(self):
        self._test_progressive_dct('9a')

    def test_progressive_dct_9b(self):
        self._test_progressive_dct('9b')

    def test_progressive_dct_9c(self):
        self._test_progressive_dct('9c')

    def test_progressive_dct_9d(self):
        self._test_progressive_dct('9d')

    def test_progressive_dct_9e(self):
        self._test_progressive_dct('9e')

    # def test_progressive_dct_turbo210(self):
    #     self._test_progressive_dct('turbo210')

    # def test_progressive_dct_mozjpeg300(self):
    #     self._test_progressive_dct('mozjpeg300')

    def test_progressive_sequential_6b(self):
        self._test_progressive_sequential('6b')

    def test_progressive_sequential_7(self):
        self._test_progressive_sequential('7')

    def test_progressive_sequential_8(self):
        self._test_progressive_sequential('8')

    def test_progressive_sequential_8a(self):
        self._test_progressive_sequential('8a')

    def test_progressive_sequential_8b(self):
        self._test_progressive_sequential('8b')

    def test_progressive_sequential_8c(self):
        self._test_progressive_sequential('8c')

    def test_progressive_sequential_8d(self):
        self._test_progressive_sequential('8d')

    def test_progressive_sequential_9(self):
        self._test_progressive_sequential('9')

    def test_progressive_sequential_9a(self):
        self._test_progressive_sequential('9a')

    def test_progressive_sequential_9b(self):
        self._test_progressive_sequential('9b')

    def test_progressive_sequential_9c(self):
        self._test_progressive_sequential('9c')

    def test_progressive_sequential_9d(self):
        self._test_progressive_sequential('9d')

    def test_progressive_sequential_9e(self):
        self._test_progressive_sequential('9e')

    # def test_progressive_sequential_turbo210(self):
    #     self._test_progressive_sequential('turbo210')

    # def test_progressive_sequential_mozjpeg300(self):
    #     self._test_progressive_sequential('mozjpeg300')


__all__ = ["TestProgressive"]
