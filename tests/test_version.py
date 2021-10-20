
import importlib
import sys
import unittest

sys.path.append('.')
import jpeglib

class TestSpatial(unittest.TestCase):
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
        

        

