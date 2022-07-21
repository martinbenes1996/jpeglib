

import logging
import numpy as np
from PIL import Image
from scipy.stats import ttest_ind
import unittest

import jpeglib


class TestPerformance(unittest.TestCase):
    def test_reading(self):
        # load and time PIL 30 times
        pil = []
        for _ in range(30):
            # timing PIL
            t_pil = jpeglib.Timer("PIL measurement")
            im = Image.open("examples/IMG_0791.jpeg")
            _ = np.array(im)
            t_pil.stop()
            pil.append(t_pil.stop())
        # load and time stegojpeg 30 times
        stego = []
        for _ in range(30):
            t_stego = jpeglib.Timer("stegojpeg measurement")
            im = jpeglib.JPEG("examples/IMG_0791.jpeg")
            _ = im.read_spatial(
                flags=['DO_FANCY_UPSAMPLING', 'DO_BLOCK_SMOOTHING']
            )
            stego.append(t_stego.stop())
        # test it isn't more than 3x slower
        max_3x_slower = ttest_ind(np.array(pil)*3, stego, alternative='less')
        logging.info(
            "performance %.2fs vs. %.2fs of PIL" % (
                np.mean(pil), np.mean(stego)
            )
        )
        self.assertGreater(max_3x_slower.pvalue, .05)

    def test_writing(self):
        pass
        # TODO


__all__ = ["TestPerformance"]
