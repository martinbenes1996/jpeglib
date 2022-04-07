
import logging
import sys
import tempfile
import unittest

sys.path.append('.')
import jpeglib

class TestMarker(unittest.TestCase):
    logger = logging.getLogger()
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='jpeg')
    def tearDown(self):
        del self.tmp
    
    def test_marker_interface(self):
        """Interface of marker."""
        self.logger.info("test_with_version")
        # create marker
        name = 'JPEG_APP0'
        content = b'abcd1234eeee'
        marker = jpeglib.Marker(
            name    = name,
            content = content,
            length  = len(content)
        )
        # attributes
        self.assertIsInstance(marker.name, str)
        self.assertEqual(marker.name, name)
        self.assertIsInstance(marker.content, bytes)
        self.assertEqual(marker.content, content)
        self.assertIsInstance(marker.length, int)
        self.assertEqual(marker.length, len(content))
        self.assertIsInstance(marker.index, int)
        self.assertEqual(marker.index, 0xE0)
    
    def test_image_marker(self):
        # read im
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # check markers
        self.assertIsInstance(im.markers, list)
        self.assertEqual(len(im.markers), 2)
        # check marker 1
        self.assertEqual(im.markers[0].name, "JPEG_APP1")
        self.assertEqual(im.markers[0].length, 2250)
        self.assertEqual(len(im.markers[0].content), 2250)
        # check marker 1
        self.assertEqual(im.markers[1].name, "JPEG_APP2")
        self.assertEqual(im.markers[1].length, 562)
        self.assertEqual(len(im.markers[1].content), 562)

__all__ = ["TestMarker"]