
import logging
import numpy as np
from PIL import Image
import tempfile
import unittest

import jpeglib


class TestShapes(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='jpeg')

    def tearDown(self):
        del self.tmp

    def test_read_dct_color(self):
        self.logger.info("test_read_dct_color")
        # read info
        im = jpeglib.read_dct("examples/IMG_0791.jpeg")
        # inner state before reading
        self.assertEqual(im.path, "examples/IMG_0791.jpeg")  # source file
        self.assertTrue(im.content is not None)
        self.assertIsInstance(im.content, bytes)
        self.assertEqual(im.height, 3024)
        self.assertEqual(im.width, 4032)
        self.assertIsInstance(im.block_dims, np.ndarray)
        self.assertEqual(im.block_dims[0, 0], im.height//8)
        self.assertEqual(im.block_dims[0, 1], im.width//8)
        self.assertEqual(im.block_dims[1, 0], im.height//8//2)
        self.assertEqual(im.block_dims[1, 1], im.width//8//2)
        self.assertEqual(im.block_dims[2, 0], im.height//8//2)
        self.assertEqual(im.block_dims[2, 1], im.width//8//2)
        self.assertIsInstance(im.samp_factor, np.ndarray)
        self.assertEqual(im.samp_factor[0, 0], 2)
        self.assertEqual(im.samp_factor[0, 1], 2)
        self.assertEqual(im.samp_factor[1, 0], 1)
        self.assertEqual(im.samp_factor[1, 1], 1)
        self.assertEqual(im.samp_factor[2, 0], 1)
        self.assertEqual(im.samp_factor[2, 1], 1)
        self.assertIsInstance(im.jpeg_color_space, jpeglib.Colorspace)
        self.assertEqual(im.num_components, 3)
        self.assertEqual(len(im.markers), 2)
        # read dct
        im.Y
        # inner state after reading
        self.assertIsInstance(im.Y, np.ndarray)
        self.assertEqual(len(im.Y.shape), 4)
        self.assertEqual(im.Y.shape[0], im.block_dims[0, 0])
        self.assertEqual(im.Y.shape[0], im.height_in_blocks(0))
        self.assertEqual(im.Y.shape[1], im.block_dims[0, 1])
        self.assertEqual(im.Y.shape[1], im.width_in_blocks(0))
        self.assertEqual(im.Y.shape[2], 8)  # block size 8^2
        self.assertEqual(im.Y.shape[3], 8)
        self.assertIsInstance(im.Cb, np.ndarray)
        self.assertEqual(len(im.Cb.shape), 4)
        self.assertEqual(im.Cb.shape[0], im.block_dims[1, 0])
        self.assertEqual(im.Cb.shape[0], im.height_in_blocks(1))
        self.assertEqual(im.Cb.shape[1], im.block_dims[1, 1])
        self.assertEqual(im.Cb.shape[1], im.width_in_blocks(1))
        self.assertEqual(im.Cb.shape[2], 8)  # block size 8^2
        self.assertEqual(im.Cb.shape[3], 8)
        self.assertIsInstance(im.Cr, np.ndarray)
        self.assertEqual(len(im.Cr.shape), 4)
        self.assertEqual(im.Cr.shape[0], im.block_dims[2, 0])
        self.assertEqual(im.Cr.shape[0], im.height_in_blocks(2))
        self.assertEqual(im.Cr.shape[1], im.block_dims[2, 1])
        self.assertEqual(im.Cr.shape[1], im.width_in_blocks(2))
        self.assertEqual(im.Cr.shape[2], 8)  # block size 8^2
        self.assertEqual(im.Cr.shape[3], 8)
        self.assertIsInstance(im.qt, np.ndarray)
        self.assertEqual(len(im.qt.shape), 3)
        # quantization tables for lumo and chroma channels
        self.assertEqual(im.qt.shape[0], 3)
        self.assertEqual(im.qt.shape[1], 8)
        self.assertEqual(im.qt.shape[2], 8)

    def test_read_spatial(self):
        self.logger.info("test_read_spatial")
        # read info
        im = jpeglib.read_spatial("examples/IMG_0791.jpeg")
        # inner state before reading
        self.assertEqual(im.path, "examples/IMG_0791.jpeg")  # source file
        self.assertTrue(im.content is not None)
        self.assertIsInstance(im.content, bytes)
        self.assertEqual(im.height, 3024)
        self.assertEqual(im.width, 4032)
        self.assertIsInstance(im.block_dims, np.ndarray)
        self.assertEqual(im.block_dims[0, 0], im.height//8)
        self.assertEqual(im.block_dims[0, 1], im.width//8)
        self.assertEqual(im.block_dims[1, 0], im.height//8//2)
        self.assertEqual(im.block_dims[1, 1], im.width//8//2)
        self.assertEqual(im.block_dims[2, 0], im.height//8//2)
        self.assertEqual(im.block_dims[2, 1], im.width//8//2)
        self.assertIsInstance(im.samp_factor, np.ndarray)
        self.assertEqual(im.samp_factor[0, 0], 2)
        self.assertEqual(im.samp_factor[0, 1], 2)
        self.assertEqual(im.samp_factor[1, 0], 1)
        self.assertEqual(im.samp_factor[1, 1], 1)
        self.assertEqual(im.samp_factor[2, 0], 1)
        self.assertEqual(im.samp_factor[2, 1], 1)
        self.assertIsInstance(im.jpeg_color_space, jpeglib.Colorspace)
        self.assertEqual(im.num_components, 3)
        self.assertEqual(len(im.markers), 2)
        # read spatial
        im.spatial
        # inner state after reading
        self.assertIsInstance(im.spatial, np.ndarray)
        self.assertEqual(len(im.spatial.shape), 3)
        self.assertEqual(im.spatial.shape[0], im.height)
        self.assertEqual(im.spatial.shape[1], im.width)
        self.assertEqual(im.spatial.shape[2], im.num_components)

    def test_read_spatial_grayscale(self):
        self.logger.info("test_read_spatial_grayscale")
        # read info
        im = jpeglib.read_spatial("examples/IMG_0791.jpeg", 'JCS_GRAYSCALE')
        # inner state before reading
        self.assertEqual(im.path, "examples/IMG_0791.jpeg")  # source file
        self.assertTrue(im.content is not None)
        self.assertIsInstance(im.content, bytes)
        self.assertEqual(im.height, 3024)
        self.assertEqual(im.width, 4032)
        self.assertIsInstance(im.block_dims, np.ndarray)
        self.assertEqual(im.block_dims[0, 0], im.height//8)
        self.assertEqual(im.block_dims[0, 1], im.width//8)
        self.assertIsInstance(im.samp_factor, np.ndarray)
        self.assertEqual(im.samp_factor[0, 0], 2)
        self.assertEqual(im.samp_factor[0, 1], 2)
        self.assertIsInstance(im.jpeg_color_space, jpeglib.Colorspace)
        self.assertEqual(im.num_components, 1)
        self.assertEqual(im.channels, 1)
        self.assertEqual(len(im.markers), 2)
        # read spatial
        im.spatial
        # inner state after reading
        self.assertIsInstance(im.spatial, np.ndarray)
        self.assertEqual(len(im.spatial.shape), 3)
        self.assertEqual(im.spatial.shape[0], im.height)
        self.assertEqual(im.spatial.shape[1], im.width)
        self.assertEqual(im.spatial.shape[2], im.channels)

    def test_read_pil(self):
        self.logger.info("test_read_pil")
        # read with PIL
        im = Image.open("examples/IMG_0791.jpeg")
        pilnpshape = np.array(im).shape
        pilsize = im.size
        # read spatial
        im = jpeglib.read_dct("examples/IMG_0791.jpeg")
        # Y
        self.assertEqual(im.Y.shape[0], pilsize[1]//8)
        self.assertEqual(im.Y.shape[1], pilsize[0]//8)
        self.assertEqual(im.Y.shape[0], pilnpshape[0]//8)
        self.assertEqual(im.Y.shape[1], pilnpshape[1]//8)
        # Cb
        self.assertEqual(im.Cb.shape[0], pilsize[1]//8//2)
        self.assertEqual(im.Cb.shape[1], pilsize[0]//8//2)
        self.assertEqual(im.Cb.shape[0], pilnpshape[0]//8//2)
        self.assertEqual(im.Cb.shape[1], pilnpshape[1]//8//2)
        # Cr
        self.assertEqual(im.Cr.shape[0], pilsize[1]//8//2)
        self.assertEqual(im.Cr.shape[1], pilsize[0]//8//2)
        self.assertEqual(im.Cr.shape[0], pilnpshape[0]//8//2)
        self.assertEqual(im.Cr.shape[1], pilnpshape[1]//8//2)
        # read dct
        im = jpeglib.read_spatial("examples/IMG_0791.jpeg")
        # spatial
        self.assertEqual(im.spatial.shape[0], pilsize[1])
        self.assertEqual(im.spatial.shape[1], pilsize[0])
        self.assertEqual(im.spatial.shape[0], pilnpshape[0])
        self.assertEqual(im.spatial.shape[1], pilnpshape[1])


__all__ = ["TestShapes"]
