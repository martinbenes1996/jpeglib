

import logging
import numpy as np
from PIL import Image
from scipy.stats import ttest_ind
import timeit
import tempfile
import unittest

import jpeglib


class TestPerformance(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')

    def tearDown(self):
        del self.tmp

    def test_reading(self):
        # load and time PIL 30 times
        res_pil = timeit.repeat(
            lambda: np.array(Image.open("examples/IMG_0791.jpeg")),
            repeat=50, number=1,
        )
        # load and time jpeglib 30 times
        res_jpeglib = timeit.repeat(
            lambda: jpeglib.read_spatial(
                "examples/IMG_0791.jpeg",
                flags=['DO_FANCY_UPSAMPLING', 'DO_BLOCK_SMOOTHING']
            ).spatial,
            repeat=50, number=1,
        )
        # test in reading, jpeglib is not more than 2x slower than PIL
        faster_than_pil = ttest_ind(np.array(res_pil)*2, res_jpeglib, alternative='greater')
        logging.info(
            "performance of reading: %.2fs vs. %.2fs of PIL" % (
                np.mean(res_jpeglib), np.mean(res_pil)
            )
        )
        self.assertLess(faster_than_pil.pvalue, .05)

    def test_writing(self):
        x = jpeglib.read_spatial("examples/IMG_0791.jpeg").spatial
        # load and time PIL 30 times
        pil = []
        res_pil = timeit.repeat(
            lambda: Image.fromarray(x).save(self.tmp.name),
            repeat=50, number=1,
        )
        # load and time jpeglib 30 times
        res_jpeglib = timeit.repeat(
            lambda: jpeglib.from_spatial(x).write_spatial(
                self.tmp.name,
                # flags=['DO_FANCY_UPSAMPLING', 'DO_BLOCK_SMOOTHING']
            ),
            repeat=50, number=1,
        )
        # test in writing, jpeglib is not slower than PIL
        faster_than_pil = ttest_ind(np.array(res_pil)*2, res_jpeglib, alternative='less')
        logging.info(
            "performance of writing: %.2fs vs. %.2fs of PIL" % (
                np.mean(res_jpeglib), np.mean(res_pil)
            )
        )
        self.assertGreater(faster_than_pil.pvalue, .05)



__all__ = ["TestPerformance"]
