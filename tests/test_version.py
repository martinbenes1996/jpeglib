"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import importlib
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import sys
import tempfile
import unittest

import jpeglib
from _defs import ALL_VERSIONS, LIBJPEG_VERSIONS


class TestVersion(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    @parameterized.expand(ALL_VERSIONS)
    def test_set_version_spatial(self, version):
        """Use a version to read and write spatial domain."""
        self.logger.info(f"test_set_version_spatial_{version}")
        jpeglib.version.set(version)
        im = jpeglib.read_spatial('tests/assets/IMG_0791.jpeg')
        im.spatial  # execute lazy loading
        im.write_spatial(self.tmp.name)
        self.assertEqual(jpeglib.version.get(), version)

    @parameterized.expand(ALL_VERSIONS)
    def test_set_version_dct(self, version):
        """Use a version to read and write DCT domain."""
        self.logger.info(f"test_set_version_dct_{version}")
        jpeglib.version.set(version)
        jpeg = jpeglib.read_dct('tests/assets/IMG_0791.jpeg')
        jpeg.Y  # execute lazy loading
        jpeg.write_dct(self.tmp.name)
        self.assertEqual(jpeglib.version.get(), version)

    def test_default_version(self):
        """Test that default version is 6b."""
        self.logger.info("test_default_version")
        # reload jpeglib
        sys.modules.pop('jpeglib._bind')
        # sys.modules.pop('jpeglib.jpeg')
        sys.modules.pop('jpeglib')
        sys.modules.pop('jpeglib.version')
        import jpeglib
        importlib.reload(jpeglib)
        # check that library is not loaded
        self.assertEqual(jpeglib.version.get(), '6b')
        # read
        im = jpeglib.read_dct('tests/assets/IMG_0791.jpeg')
        im.load()
        # check default version
        self.assertEqual(jpeglib.version.get(), '6b')

    def test_with_version(self):
        """Test with statement correctly setting and resetting the version."""
        # default version
        self.assertEqual(jpeglib.version.get(), '6b')
        # version changes in the block
        with jpeglib.version('9e'):
            self.assertEqual(jpeglib.version.get(), '9e')
        # back to default version
        self.assertEqual(jpeglib.version.get(), '6b')

    def test_versions(self):
        """Test listing available versions."""
        versions = jpeglib.version.versions()
        for (v,) in ALL_VERSIONS:
            self.assertIn(v, versions)

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_libjpeg_testimages(self, version):
        """Test on test images from libjpeg.

        Test cases from libjpeg Makefile:
            $ ./djpeg -dct int -ppm -outfile testout.ppm  testorig.jpg
            $ cmp testimg.ppm testout.ppm
            $ ./djpeg -dct int -bmp -colors 256 -outfile testout.bmp  testorig.jpg
            $ cmp testimg.bmp testout.bmp
            $ ./cjpeg -dct int -outfile testout.jpg  testimg.ppm
            $ cmp testimg.jpg testout.jpg
            $ ./djpeg -dct int -ppm -outfile testoutp.ppm testprog.jpg
            $ cmp testimg.ppm testoutp.ppm
            $ ./cjpeg -dct int -progressive -opt -outfile testoutp.jpg testimg.ppm
            $ cmp testimgp.jpg testoutp.jpg
            $ ./jpegtran -outfile testoutt.jpg testprog.jpg
            $ cmp testorig.jpg testoutt.jpg
        """
        self.logger.info(f"test_libjpeg_testimages_{version}")
        jpeglib.version.set(version)
        if sys.platform.startswith("win"):
            logging.warning('TODO: investigate mismatch on Windows vs. Linux')
            return

        # test original
        im_ppm = Image.open(f'tests/assets/images-{version}/testimg.ppm')
        rgb_ppm = np.array(im_ppm)
        im = jpeglib.read_spatial(f'tests/assets/images-{version}/testorig.jpg')
        np.testing.assert_array_almost_equal(im.spatial, rgb_ppm)

        # TODO: fix quantization
        # # test 256 colors
        # im_bmp = Image.open("tests/assets/images-6b/testimg.bmp")
        # bmp_palette = np.array(im_bmp.getpalette()).reshape(-1,3)#[:,::-1]
        # print("Before call")
        # print(bmp_palette[:6])
        # rgb_bmp = np.array([[bmp_palette[i] for i in row] for row in np.array(im_bmp)])  # noqa: E501
        # im = jpeglib.JPEG(f'tests/assets/images-{version}/testorig.jpg')
        # rgb = im.read_spatial(out_color_space='JCS_RGB', colormap=bmp_palette, flags=['QUANTIZE_COLORS'])  # noqa: E501
        # rgb = np.array([[bmp_palette[i] for i in row] for row in np.array(rgb)])  # noqa: E501
        # np.testing.assert_array_equal(rgb, rgb_bmp)
        # print(rgb_bmp.shape)

        # compress
        jpeg = jpeglib.read_dct(f'tests/assets/images-{version}/testimg.jpg')
        im_ppm = Image.open(f"tests/assets/images-{version}/testimg.ppm")
        rgb_ppm = np.array(im_ppm)
        jpeglib.from_spatial(rgb_ppm).write_spatial(self.tmp.name)
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_equal(jpeg.Cr, jpeg2.Cr)
        np.testing.assert_equal(jpeg.qt, jpeg2.qt)

        # progressive compression
        jpeglib.from_spatial(
            spatial=rgb_ppm,
        ).write_spatial(self.tmp.name, flags=['+PROGRESSIVE_MODE'])
        jpeg = jpeglib.read_dct(f'tests/assets/images-{version}/testimgp.jpg')
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_array_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_array_equal(jpeg.Cr, jpeg2.Cr)
        np.testing.assert_array_equal(jpeg.qt, jpeg2.qt)

        # no difference between progressive and sequential
        jpeg = jpeglib.read_dct(f'tests/assets/images-{version}/testprog.jpg')
        jpeg2 = jpeglib.read_dct(f'tests/assets/images-{version}/testorig.jpg')
        np.testing.assert_array_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_array_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_array_equal(jpeg.Cr, jpeg2.Cr)
        np.testing.assert_array_equal(jpeg.qt, jpeg2.qt)

    @parameterized.expand([
        ['libjpeg6b'],
        ['6c'],
        ['turbo209'],
        ['nonexistent']
    ])
    def test_bad_version(self, version):
        # bad version set
        try:
            jpeglib.version.set(version)
        except RuntimeError as e:
            self.assertEqual(str(e), f'version "{version}" not found, was the package compiled correctly?')
        # bad with version
        try:
            with jpeglib.version(version):
                pass
        except RuntimeError as e:
            self.assertEqual(str(e), f'version "{version}" not found, was the package compiled correctly?')

    @parameterized.expand(ALL_VERSIONS)
    def test_version_custom_QT(self, version):
        """Test for bug using custom QT in 9e (thows sigsegv)."""

        # Create a random 8x8 image
        rng = np.random.default_rng(12345)
        spatial = rng.integers(low=0, high=256, size=(8, 8, 1), dtype='uint8')
        assert spatial.min() >= 0
        assert spatial.max() <= 255

        # Create a jpeglib object
        im = jpeglib.from_spatial(spatial)

        # Create a custom quantization table
        qt = np.ones((1, 8, 8))

        # Using version "6b" or "mozjpeg403" works, but "9e" exits with code 139 (interrupted by signal 11: SIGSEGV)
        with jpeglib.version(version):
            im.write_spatial(self.tmp.name, qt=qt)


__all__ = ["TestVersion"]
