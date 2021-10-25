
import numpy as np
from PIL import Image
import sys
import unittest

sys.path.append('.')
import jpeglib


class TestShapes(unittest.TestCase):
    def test_read_direct(self):

        # read info
        im = jpeglib.JPEG("examples/IMG_0791.jpeg")
        # inner state
        self.assertIsNotNone(jpeglib.JPEG.cjpeglib) # dylib object
        self.assertEqual(im.srcfile, "examples/IMG_0791.jpeg") # source file
        self.assertEqual(im.dct_channels, 3) # color channels
        self.assertEqual(im.channels, 3) # color channels

        # read image
        Y,CbCr,qt = im.read_dct()
        
        # test Y shapes
        self.assertIsInstance(Y, np.ndarray)
        self.assertEqual(len(Y.shape), 5)
        self.assertEqual(Y.shape[0], 1) # Y contains one color channel
        self.assertEqual(Y.shape[1], im.dct_shape[0][0])
        self.assertEqual(Y.shape[2], im.dct_shape[0][1])
        self.assertEqual(Y.shape[3], 8) # block size 8^2
        self.assertEqual(Y.shape[4], 8)
        # test CbCr
        self.assertIsInstance(CbCr, np.ndarray)
        self.assertEqual(len(CbCr.shape), 5)
        self.assertEqual(CbCr.shape[0], 2) # CbCr contains two color channels
        self.assertEqual(CbCr.shape[1], im.dct_shape[1][0])
        self.assertEqual(CbCr.shape[2], im.dct_shape[1][1])
        self.assertEqual(CbCr.shape[3], 8) # block size 8^2
        self.assertEqual(CbCr.shape[4], 8)
        # test qt
        self.assertIsInstance(qt, np.ndarray)
        self.assertEqual(len(qt.shape), 3)
        self.assertEqual(qt.shape[0], 2) # quantization tables for lumo and chroma channels
        self.assertEqual(qt.shape[1], 8)
        self.assertEqual(qt.shape[2], 8)

        # read image
        data = im.read_spatial()
        # test spatial
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(len(data.shape), 3)
        self.assertEqual(data.shape[0], im.shape[1])
        self.assertEqual(data.shape[1], im.shape[0])
        self.assertEqual(data.shape[2], 3)

        # cleanup
        im.close()

        
    def test_read_pil(self):
        
        # read with PIL
        with Image.open("examples/IMG_0791.jpeg") as im:
            pilnpshape = np.array(im).shape
            pilsize = im.size
        # read image
        im = jpeglib.JPEG("examples/IMG_0791.jpeg")
        Y,CbCr,qt = im.read_dct()
        data = im.read_spatial()

        # Y
        self.assertEqual(Y.shape[1], pilsize[0]/8)
        self.assertEqual(Y.shape[2], pilsize[1]/8)
        self.assertEqual(Y.shape[1], pilnpshape[1]/8)
        self.assertEqual(Y.shape[2], pilnpshape[0]/8)
        # CbCr
        self.assertEqual(CbCr.shape[1], pilsize[0]/8/2)
        self.assertEqual(CbCr.shape[2], pilsize[1]/8/2)
        self.assertEqual(CbCr.shape[1], pilnpshape[1]/8/2)
        self.assertEqual(CbCr.shape[2], pilnpshape[0]/8/2)
        # spatial
        self.assertEqual(data.shape[0], pilsize[1])
        self.assertEqual(data.shape[1], pilsize[0])
        self.assertEqual(data.shape[0], pilnpshape[0])
        self.assertEqual(data.shape[1], pilnpshape[1])
        # cleanup
        im.close()


            
__all__ = ["TestShapes"]