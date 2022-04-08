import unittest
import tempfile
import logging
import sys

sys.path.append('.')


class TestProgressive(unittest.TestCase):

    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='jpeg')

    def tearDown(self):
        del self.tmp

    def test_read_progressive_flag(self):
        import jpeglib

        self.logger.info("test_read_progressive_flag")

        im = jpeglib.read_spatial("examples/images-6b/testprog.jpg")
        self.assertTrue(im.progressive_mode)

        im = jpeglib.read_spatial("examples/images-6b/testimg.jpg")
        self.assertFalse(im.progressive_mode)


__all__ = ["TestProgressive"]
