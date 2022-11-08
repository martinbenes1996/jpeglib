
import logging
import numpy as np
import os
from scipy import fftpack
from scipy import fft
import tempfile
import unittest

import jpeglib


class TestOps(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    def test_dct2d(self):
        # generate spatial
        np.random.seed(12345)
        x = np.random.randint(0,256,(8,8)).astype('float64')
        xb = np.expand_dims(x, (0,1))
        # jpeglib forward dct
        yb1 = jpeglib.ops.forward_dct(xb)[0,0]
        # scipy forward dct
        yb2 = fftpack.dct(fftpack.dct(x.T, norm='ortho').T, norm='ortho')
        # check equal
        np.testing.assert_array_almost_equal(yb1, yb2)

    def test_idct2d(self):
        # generate spatial
        np.random.seed(12345)
        x = np.random.randint(0,256,(8,8)).astype('float64')
        xb = np.expand_dims(x, (0,1))
        # jpeglib forward dct
        yb1 = jpeglib.ops.backward_dct(xb)[0,0]
        # scipy forward dct
        yb2 = fftpack.idct(fftpack.idct(x.T, norm='ortho').T, norm='ortho')
        # check equal
        np.testing.assert_array_almost_equal(yb1, yb2)