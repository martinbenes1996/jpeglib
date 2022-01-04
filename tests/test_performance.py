

import numpy as np
from scipy.stats import ttest_ind
from PIL import Image
import sys
import unittest

sys.path.append('.')
import jpeglib


class TestPerformance(unittest.TestCase):
    def test_reading(self):
        # load and time PIL 30 times
        pil = []
        for _ in range(30):
            # timing PIL
            t_pil = jpeglib.Timer("PIL measurement")
            im = Image.open("examples/IMG_0791.jpeg")
            x = np.array(im)
            t_pil.stop()
            pil.append(t_pil.stop())
        # load and time stegojpeg 30 times
        stego = []
        for _ in range(30):
            t_stego = jpeglib.Timer("stegojpeg measurement")
            im = jpeglib.JPEG("examples/IMG_0791.jpeg")
            x = im.read_spatial(
                flags=['DO_FANCY_UPSAMPLING','DO_BLOCK_SMOOTHING']
            )
            stego.append(t_stego.stop())
        # test it isn't more than 3x slower
        max_3x_slower = ttest_ind(np.array(pil)*3, stego, alternative='less')
        self.assertGreater(max_3x_slower.pvalue, .05)
    
    def test_writing(self):
        pass
        # TODO
        

__all__ = ["TestPerformance"]
