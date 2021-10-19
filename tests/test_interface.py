
import sys
import unittest

sys.path.append('.')
import jpeglib


class TestInterface(unittest.TestCase):

    def test_read_with(self):
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()
            
    def test_rgb(self):
        im = jpeglib.JPEG("examples/IMG_0791.jpeg")
        Y,CbCr,qt = im.read_dct()
        im.close()
            

__all__ = ["TestInterface"]