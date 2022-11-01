

import logging
import numpy as np
from PIL import Image
from scipy.stats import ttest_ind, ttest_1samp
import timeit
import tempfile
import unittest

import jpeglib


class TestPerformance(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')

    def tearDown(self):
        self.tmp.close()
        del self.tmp

    def test_reading(self):
        self.logger.info("test_reading")
        # load and time jpeglib 50 times
        jpeglib.version.set('turbo210')
        res_jpeglib = timeit.repeat(
            lambda: jpeglib.read_spatial(
                "examples/IMG_0791.jpeg",
                flags=['DO_FANCY_UPSAMPLING', 'DO_BLOCK_SMOOTHING']
            ).spatial,
            repeat=50, number=1,
        )
        # test in reading, jpeglin is faster than 300ms
        faster_than_300ms = ttest_1samp(res_jpeglib, .3, alternative='less')
        logging.info(
            "performance of reading: %.2fs" % (
                np.mean(res_jpeglib)
            )
        )
        self.assertLess(faster_than_300ms.pvalue, .05)

    def test_writing(self):
        self.logger.info("test_writing")
        x = jpeglib.read_spatial("examples/IMG_0791.jpeg").spatial
        jpeglib.version.set('turbo210')
        # load and time jpeglib 50 times
        res_jpeglib = timeit.repeat(
            lambda: jpeglib.from_spatial(x).write_spatial(
                self.tmp.name,
            ),
            repeat=50, number=1,
        )
        # test in writing, jpeglib is faster than 300ms
        faster_than_300ms = ttest_1samp(res_jpeglib, .3, alternative='less')
        logging.info(
            "performance of writing: %.2fs" % (
                np.mean(res_jpeglib)
            )
        )
        self.assertLess(faster_than_300ms.pvalue, .05)



__all__ = ["TestPerformance"]
