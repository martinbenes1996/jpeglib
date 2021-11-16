
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy.stats import kstest
import shutil
import sys
import unittest

sys.path.append('.')
import jpeglib

class TestVersion(unittest.TestCase):
    def setUp(self):
        try: shutil.rmtree("tmp")
        except: pass
        finally: os.mkdir("tmp")
    def tearDown(self):
        shutil.rmtree("tmp")

    def test_6b(self):
        jpeglib.version.set('6b')
        with jpeglib.JPEG('examples/IMG_0791.jpeg') as im:
            Y,CbCr,qt = im.read_dct()
            rgb = im.read_spatial()
        self.assertEqual(jpeglib.version.get(), '6b')

    def test_8d(self):
        jpeglib.version.set('8d')
        with jpeglib.JPEG('examples/IMG_0791.jpeg') as im:
            Y,CbCr,qt = im.read_dct()
            rgb = im.read_spatial()
        self.assertEqual(jpeglib.version.get(), '8d')
    
    def test_6(self):
        jpeglib.version.set('6')
        self.assertEqual(jpeglib.version.get(), '6b')
    def test_8(self):
        jpeglib.version.set('8')
        self.assertEqual(jpeglib.version.get(), '8d')

    def test_default_version(self):
        # reload jpeglib
        sys.modules.pop('jpeglib._bind')
        sys.modules.pop('jpeglib.jpeg')
        importlib.reload(jpeglib)
        # check that library is not loaded
        self.assertIsNone(jpeglib.version.get())
        # read
        im = jpeglib.JPEG('examples/IMG_0791.jpeg')
        Y,CbCr,qt = im.read_dct()
        # check default version
        self.assertEqual(jpeglib.version.get(), '6b')
    
    def _test_libjpeg_images(self, version):
        """Test on test images from libjpeg."""
        jpeglib.version.set(version)
        # test original
        im_ppm = Image.open(f'examples/images-{version}/testimg.ppm')
        rgb_ppm = np.array(im_ppm)
        im = jpeglib.JPEG(f'examples/images-{version}/testorig.jpg')
        rgb = im.read_spatial(out_color_space='JCS_RGB', flags=['DO_FANCY_UPSAMPLING'])
        np.testing.assert_array_equal(rgb, rgb_ppm)
        
        # # test 256 colors
        #im_bmp = Image.open("examples/images-6b/testimg.bmp")
        #bmp_palette = np.array(im_bmp.getpalette()).reshape(-1,3)#[:,::-1]
        # print("Before call")
        #print(bmp_palette[:6])
        #rgb_bmp = np.array([[bmp_palette[i] for i in row] for row in np.array(im_bmp)])
        #im = jpeglib.JPEG(f'examples/images-{version}/testorig.jpg')
        #rgb = im.read_spatial(out_color_space='JCS_RGB', colormap=bmp_palette, flags=['QUANTIZE_COLORS'])
        #rgb = np.array([[bmp_palette[i] for i in row] for row in np.array(rgb)])
        #np.testing.assert_array_equal(rgb, rgb_bmp)
        #print(rgb_bmp.shape)
        
        #plt.imshow(rgb)
        #D = np.abs((rgb_bmp - rgb).astype(np.int8)).mean(axis=2)
        #plt.imshow(D, cmap='gray')
        #plt.show()
        #return

        # compress
        kw = {'flags': ['DO_FANCY_UPSAMPLING','DO_BLOCK_SMOOTHING']}
        im_ppm = Image.open(f"examples/images-{version}/testimg.ppm")
        rgb_ppm = np.array(im_ppm)
        im = jpeglib.JPEG()#f'examples/images-{version}/testimg.jpg')
        im.write_spatial('tmp/output.jpeg', rgb_ppm, **kw)
        im_compressed = jpeglib.JPEG('tmp/output.jpeg')
        rgb_compressed = im_compressed.read_spatial(**kw)
        im = jpeglib.JPEG(f'examples/images-{version}/testimg.jpg')
        rgb = im.read_spatial(**kw)
        # test same histogram
        hist1,_ = np.histogram(rgb, bins=256, range=(0,256), density=True)
        hist2,_ = np.histogram(rgb_compressed, bins=256, range=(0,256), density=True)
        self.assertGreaterEqual(kstest(hist2, hist1).pvalue, .05)

        # load dct
        im_prog = jpeglib.JPEG(f'examples/images-{version}/testprog.jpg')
        print("read_dct")
        Y,CbCr,qt = im_prog.read_dct()
        print("write_dct")
        im_prog.write_dct("tmp/output.jpeg", Y, CbCr, qt)
        print("open")
        im = jpeglib.JPEG("tmp/output.jpeg")
        print("read_dct")
        Y2,CbCr2,qt2 = im.read_dct()
        print("test")
        np.testing.assert_array_equal(Y, Y2)
        np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(qt, qt2)

        # load progressive image
        im_seq = jpeglib.JPEG(f'examples/images-{version}/testimg.jpg')
        rgb_seq = im_seq.read_spatial(out_color_space='JCS_RGB')
        im_p = jpeglib.JPEG(f'examples/images-{version}/testimgp.jpg')
        rgb_p = im_p.read_spatial(out_color_space='JCS_RGB', flags=['PROGRESSIVE_MODE'])
        np.testing.assert_array_almost_equal(rgb_seq, rgb_p)

    # def test_libjpeg_images_6b(self):
    #     """Test on test images from libjpeg 6b."""
    #     self._test_libjpeg_images("6b")
    def test_libjpeg_images_8d(self):
        """Test on test images from libjpeg 8d."""
        self._test_libjpeg_images("8d")
        
__all__ = ["TestVersion"]