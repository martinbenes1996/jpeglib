"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import logging
import numpy as np
import os
from scipy.stats import ttest_1samp
import timeit
import tempfile
import unittest

import jpeglib


class TestPerformance(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    def test_reading(self):
        """Test reading is statistically significantly faster than 300ms."""
        self.logger.info("test_reading")
        # load and time jpeglib 50 times
        jpeglib.version.set('turbo210')
        res_jpeglib = timeit.repeat(
            lambda: jpeglib.read_spatial(
                "tests/assets/IMG_0791.jpeg",
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
        """Test writing is statistically significantly faster than 500ms."""
        self.logger.info("test_writing")
        x = jpeglib.read_spatial("tests/assets/IMG_0791.jpeg").spatial
        jpeglib.version.set('turbo210')
        # load and time jpeglib 50 times
        res_jpeglib = timeit.repeat(
            lambda: jpeglib.from_spatial(x).write_spatial(
                self.tmp.name,
            ),
            repeat=50, number=1,
        )
        # test in writing, jpeglib is faster than 500ms
        faster_than_500ms = ttest_1samp(res_jpeglib, .5, alternative='less')
        logging.info(
            "performance of writing: %.2fs" % (
                np.mean(res_jpeglib)
            )
        )
        self.assertLess(faster_than_500ms.pvalue, .05)


__all__ = ["TestPerformance"]
