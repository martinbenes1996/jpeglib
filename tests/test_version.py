
import importlib
import logging
import numpy as np
from PIL import Image
import sys
import tempfile
import unittest

import jpeglib


class TestVersion(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')

    def tearDown(self):
        del self.tmp

    def _read_image(self):
        jpeglib.read_dct('examples/IMG_0791.jpeg').Y
        jpeglib.read_spatial('examples/IMG_0791.jpeg').spatial

    def _test_version_set(self, version):
        self.logger.info(f"test_version_set_{version}")
        jpeglib.version.set(version)
        self._read_image()
        self.assertEqual(jpeglib.version.get(), version)

    def test_version_set_6b(self):
        self._test_version_set('6b')
    def test_version_set_7(self):
        self._test_version_set('7')
    def test_version_set_8(self):
        self._test_version_set('8')
    def test_version_set_8a(self):
        self._test_version_set('8a')
    def test_version_set_8b(self):
        self._test_version_set('8b')
    def test_version_set_8c(self):
        self._test_version_set('8c')
    def test_version_set_8d(self):
        self._test_version_set('8d')
    def test_version_set_9(self):
        self._test_version_set('9')
    def test_version_set_9a(self):
        self._test_version_set('9a')
    def test_version_set_9b(self):
        self._test_version_set('9b')
    def test_version_set_9c(self):
        self._test_version_set('9c')
    def test_version_set_9d(self):
        self._test_version_set('9d')
    def test_version_set_9e(self):
        self._test_version_set('9e')

    def test_version_set_turbo120(self):
        self._test_version_set('turbo120')
    def test_version_set_turbo130(self):
        self._test_version_set('turbo130')
    def test_version_set_turbo140(self):
        self._test_version_set('turbo140')
    def test_version_set_turbo150(self):
        self._test_version_set('turbo150')
    def test_version_set_turbo200(self):
        self._test_version_set('turbo200')
    def test_version_set_turbo210(self):
        self._test_version_set('turbo210')

    def test_version_set_mozjpeg101(self):
        self._test_version_set('mozjpeg101')
    def test_version_set_mozjpeg201(self):
        self._test_version_set('mozjpeg201')
    def test_version_set_mozjpeg300(self):
        self._test_version_set('mozjpeg300')
    def test_version_set_mozjpeg403(self):
        self._test_version_set('mozjpeg403')

    def test_default_version(self):
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
        im = jpeglib.read_dct('examples/IMG_0791.jpeg')
        im.load()
        # check default version
        self.assertEqual(jpeglib.version.get(), '6b')

    def _test_libjpeg_testimages(self, version):
        """Test on test images from libjpeg."""
        self.logger.info(f"test_libjpeg_testimages_{version}")
        jpeglib.version.set(version)
        # test original
        im_ppm = Image.open(f'examples/images-{version}/testimg.ppm')
        rgb_ppm = np.array(im_ppm)
        im = jpeglib.read_spatial(f'examples/images-{version}/testorig.jpg')
        np.testing.assert_array_almost_equal(im.spatial, rgb_ppm)

        # # test 256 colors
        # im_bmp = Image.open("examples/images-6b/testimg.bmp")
        # bmp_palette = np.array(im_bmp.getpalette()).reshape(-1,3)#[:,::-1]
        # print("Before call")
        # print(bmp_palette[:6])
        # rgb_bmp = np.array([[bmp_palette[i] for i in row] for row in np.array(im_bmp)])  # noqa: E501
        # im = jpeglib.JPEG(f'examples/images-{version}/testorig.jpg')
        # rgb = im.read_spatial(out_color_space='JCS_RGB', colormap=bmp_palette, flags=['QUANTIZE_COLORS'])  # noqa: E501
        # rgb = np.array([[bmp_palette[i] for i in row] for row in np.array(rgb)])  # noqa: E501
        # np.testing.assert_array_equal(rgb, rgb_bmp)
        # print(rgb_bmp.shape)

        # plt.imshow(rgb)
        # D = np.abs((rgb_bmp - rgb).astype(np.int8)).mean(axis=2)
        # plt.imshow(D, cmap='gray')
        # plt.show()
        # return

        # compress
        # kw = {'flags': ['DO_FANCY_UPSAMPLING','DO_BLOCK_SMOOTHING']}
        # im_ppm = Image.open(f"examples/images-{version}/testimg.ppm")
        # rgb_ppm = np.array(im_ppm)
        # im = jpeglib.JPEG()#f'examples/images-{version}/testimg.jpg')
        # im.write_spatial('tmp/output.jpeg', rgb_ppm, **kw)
        # im_compressed = jpeglib.JPEG('tmp/output.jpeg')
        # rgb_compressed = im_compressed.read_spatial(**kw)
        # im = jpeglib.JPEG(f'examples/images-{version}/testimg.jpg')
        # rgb = im.read_spatial(**kw)
        # # test same histogram
        # hist1,_ = np.histogram(rgb, bins=256, range=(0,256), density=True)
        # hist2,_ = np.histogram(rgb_compressed, bins=256, range=(0,256), density=True)  # noqa: E501
        # self.assertGreaterEqual(kstest(hist2, hist1).pvalue, .05)

        im_prog = jpeglib.read_spatial(
            f'examples/images-{version}/testprog.jpg',
            flags=[
                '+PROGRESSIVE_MODE',
                '+DO_FANCY_UPSAMPLING',
                '+DO_BLOCK_SMOOTHING'
            ]
        )
        # rgb_pil = np.array(Image.open(
        _ = np.array(Image.open(
            f'examples/images-{version}/testprog.jpg'
        ))
        # np.testing.assert_array_almost_equal(im_prog.spatial, rgb_pil)  # TODO: Nora

        # load dct - to fix
        im = jpeglib.read_dct(f'examples/images-{version}/testprog.jpg')
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        # im_prog = jpeglib.JPEG()
        # im_prog.write_dct(Y, CbCr, self.tmp.name, qt=qt)
        # im = jpeglib.JPEG(self.tmp.name)
        # Y2, CbCr2, qt2 = im.read_dct()
        np.testing.assert_array_equal(im.Y, im2.Y)
        # D = np.abs((CbCr.astype(np.int) - CbCr2.astype(np.int)))
        # print((D != 0).sum()) # 1073 mismatches
        # np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(im.qt, im2.qt)

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

    def test_libjpeg_testimages_6b(self):
        """Test on test images from libjpeg 6b."""
        self._test_libjpeg_testimages("6b")

    def test_libjpeg_testimages_7(self):
        """Test on test images from libjpeg 7."""
        self._test_libjpeg_testimages("7")

    def test_libjpeg_testimages_8(self):
        """Test on test images from libjpeg 8."""
        self._test_libjpeg_testimages("8")

    def test_libjpeg_testimages_8a(self):
        """Test on test images from libjpeg 8a."""
        self._test_libjpeg_testimages("8a")

    def test_libjpeg_testimages_8b(self):
        """Test on test images from libjpeg 8b."""
        self._test_libjpeg_testimages("8b")

    def test_libjpeg_testimages_8c(self):
        """Test on test images from libjpeg 8c."""
        self._test_libjpeg_testimages("8c")

    def test_libjpeg_testimages_8d(self):
        """Test on test images from libjpeg 8d."""
        self._test_libjpeg_testimages("8d")

    def test_libjpeg_testimages_9(self):
        """Test on test images from libjpeg 9."""
        self._test_libjpeg_testimages("9")

    def test_libjpeg_testimages_9a(self):
        """Test on test images from libjpeg 9a."""
        self._test_libjpeg_testimages("9a")

    def test_libjpeg_images_9b(self):
        """Test on test images from libjpeg 9b."""
        self._test_libjpeg_testimages("9b")

    def test_libjpeg_images_9c(self):
        """Test on test images from libjpeg 9c."""
        self._test_libjpeg_testimages("9c")

    def test_libjpeg_images_9d(self):
        """Test on test images from libjpeg 9d."""
        self._test_libjpeg_testimages("9d")

    def test_libjpeg_images_9e(self):
        """Test on test images from libjpeg 9e."""
        self._test_libjpeg_testimages("9e")

    # TODO
    # def test_libjpeg_images_turbo210(self):
    #    """Test on test images from libjpeg-turbo 2.1.0."""
    #    self._test_libjpeg_images("turbo210")
    # def test_libjpeg_images_mozjpeg403(self):
    #    """Test on test images from mozjpeg 4.0.3."""
    #    self._test_libjpeg_images("mozjpeg403")


__all__ = ["TestVersion"]
