
import sys
import unittest

sys.path.append('.')
import jpeglib


class TestSpatial(unittest.TestCase):
    def test_6b(self):
        jpeglib.set_libjpeg_version('6b')
        with jpeglib.JPEG('examples/IMG_0791.jpeg') as im:
            Y,CbCr,qt = im.read_dct()
            rgb = im.read_spatial()
            

    def test_8d(self):
        jpeglib.set_libjpeg_version('8d')
        with jpeglib.JPEG('examples/IMG_0791.jpeg') as im:
            Y,CbCr,qt = im.read_dct()
            rgb = im.read_spatial()