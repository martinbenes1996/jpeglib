
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
    
    def test_with_version(self):
        """Test with statement for version."""
        # default version
        jpeglib.version.set('6b')
        self.assertEqual(jpeglib.version.get(), '6b')
        # block with new version
        with jpeglib.version('8d'):
            self.assertEqual(jpeglib.version.get(), '8d')
            # block with new version
            with jpeglib.version('9d'):
                self.assertEqual(jpeglib.version.get(), '9d')
            # back to 8d
            self.assertEqual(jpeglib.version.get(), '8d')
        # back to 6b
        self.assertEqual(jpeglib.version.get(), '6b')
            

__all__ = ["TestInterface"]