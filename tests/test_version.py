
import importlib
import numpy as np
from PIL import Image
import sys
import unittest

sys.path.append('.')
import jpeglib

class TestVersion(unittest.TestCase):
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
        im_ppm = Image.open("examples/images-6b/testimg.ppm")
        rgb_ppm = np.array(im_ppm)
        im = jpeglib.JPEG(f'examples/images-{version}/testorig.jpg')
        rgb = im.read_spatial(out_color_space='JCS_RGB', flags=['DO_FANCY_UPSAMPLING'])
        np.testing.assert_array_equal(rgb, rgb_ppm)
        
        # test 256 colors
        im_bmp = Image.open("examples/images-6b/testimg.bmp")
        rgb_bmp = np.array(im_bmp.convert('RGB'))
        bmp_palette = np.array(list(im_bmp.palette.colors.keys()))
        print(im_bmp.palette.colors)
        print(bmp_palette)
        #print(im_bmp.palette.colors)
        # '_palette', 'colors', 'copy', 'dirty', 'getcolor', 'getdata', 'mode', 'palette', 'rawmode', 'save', 'tobytes', 'tostring']
        im = jpeglib.JPEG(f'examples/images-{version}/testorig.jpg')
        rgb = im.read_spatial(out_color_space='JCS_RGB', colormap=np.array(list(im_bmp.palette.colors.keys())), flags=['QUANTIZE_COLORS'])
        #import matplotlib.pyplot as plt
        #plt.imshow(rgb)
        #plt.show()
        #return
        #plt.imshow((rgb - rgb_bmp).astype(np.int))
        #plt.show()
        np.testing.assert_array_equal(rgb, rgb_bmp)
        
        return
        # load test image
        im_seq = jpeglib.JPEG(f'examples/images-{version}/testimg.jpg')
        rgb_seq = im_seq.read_spatial(out_color_space='JCS_RGB')
        # load progressive image
        im_p = jpeglib.JPEG(f'examples/images-{version}/testimgp.jpg')
        rgb_p = im_p.read_spatial(out_color_space='JCS_RGB', flags=['PROGRESSIVE_MODE'])
        # load progressive image
        im_prog = jpeglib.JPEG(f'examples/images-{version}/testprog.jpg')
        rgb_prog = im_prog.read_spatial(out_color_space='JCS_RGB', flags=['PROGRESSIVE_MODE','ENABLE_2PASS_QUANT'])
        # check all equal
        np.testing.assert_array_equal(rgb_orig, rgb_seq)
        np.testing.assert_array_equal(rgb_seq, rgb_p)
        np.testing.assert_array_equal(rgb_seq, rgb_prog)
        
        #import matplotlib.pyplot as plt
        #plt.imshow((rgb_prog == rgb_ppm).astype(np.int))
        #plt.show()
    def test_libjpeg_images_6b(self):
        """Test on test images from libjpeg 6b."""
        self._test_libjpeg_images("6b")
    #def test_libjpeg_images_8d(self):
    #    """Test on test images from libjpeg 6b."""
    #    self._test_libjpeg_images("8d")
    
        
__all__ = ["TestVersion"]