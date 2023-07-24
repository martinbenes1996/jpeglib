"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import logging
import os
import tempfile
import unittest

import jpeglib


class TestMarker(unittest.TestCase):
    logger = logging.getLogger()

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    def test_marker_jpeg_app0(self):
        """Interface of marker."""
        self.logger.info("test_marker_interface")
        # create marker
        name = 'JPEG_APP0'
        content = b'abcd1234eeee'
        marker = jpeglib.Marker(
            type=jpeglib.MarkerType[name],
            content=content,
            length=len(content)
        )
        # attributes
        self.assertEqual(str(marker.type), name)
        self.assertIsInstance(marker.content, bytes)
        self.assertEqual(marker.content, content)
        self.assertIsInstance(marker.length, int)
        self.assertEqual(marker.length, len(content))
        self.assertIsInstance(int(marker.type), int)
        self.assertEqual(int(marker.type), 0xE0)

    def test_marker_jpeg_app5(self):
        """Interface of marker."""
        self.logger.info("test_marker_interface")
        # create marker
        name = 'JPEG_APP5'
        content = b'abcd1234eeee'
        marker = jpeglib.Marker(
            type=jpeglib.MarkerType[name],
            content=content,
            length=len(content)
        )
        # attributes
        self.assertIsInstance(str(marker.type), str)
        self.assertEqual(str(marker.type), name)
        self.assertIsInstance(int(marker.type), int)
        self.assertEqual(int(marker.type), 0xE0 + 5)
        self.assertIsInstance(marker.content, bytes)
        self.assertEqual(marker.content, content)
        self.assertIsInstance(marker.length, int)
        self.assertEqual(marker.length, len(content))

    def test_image_marker(self):
        self.logger.info("test_image_marker")
        # read im
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # check markers
        self.assertIsInstance(im.markers, list)
        self.assertEqual(len(im.markers), 2)
        self.assertIsInstance(str(im.markers), str)
        # check marker 1
        self.assertEqual(str(im.markers[0].type), "JPEG_APP1")
        self.assertEqual(im.markers[0].length, 2250)
        self.assertEqual(len(im.markers[0].content), 2250)
        # check marker 1
        self.assertEqual(str(im.markers[1].type), "JPEG_APP2")
        self.assertEqual(im.markers[1].length, 562)
        self.assertEqual(len(im.markers[1].content), 562)


__all__ = ["TestMarker"]
