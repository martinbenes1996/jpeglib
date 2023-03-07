"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import logging
import numpy as np
import os
from parameterized import parameterized
import tempfile
import unittest

import jpeglib


class TestNotation(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    @parameterized.expand([
        [[4, 4, 4], [[1, 1], [1, 1], [1, 1]]],
        [[4, 4, 0], [[1, 2], [1, 1], [1, 1]]],
        [[4, 2, 2], [[2, 1], [1, 1], [1, 1]]],
        [[4, 2, 0], [[2, 2], [1, 1], [1, 1]]],
        [[4, 1, 1], [[4, 1], [1, 1], [1, 1]]],
        [[4, 1, 0], [[4, 2], [1, 1], [1, 1]]],
        [[3, 3, 3], [[1, 1], [1, 1], [1, 1]]],
        [[3, 3, 0], [[1, 2], [1, 1], [1, 1]]],
        [[3, 1, 1], [[3, 1], [1, 1], [1, 1]]],
        [[3, 1, 0], [[3, 2], [1, 1], [1, 1]]],
    ])
    def test_Jab(self, Jab, samp_factor):
        """Test of J:a:b to samp factor convertor."""
        self.logger.info(f'test_Jab({Jab})')
        np.testing.assert_array_equal(
            # convert Jab to factors
            jpeglib.Jab_to_factors(Jab),
            # reference
            samp_factor
        )

    @parameterized.expand([
        ['4:4:4', [[1, 1], [1, 1], [1, 1]]],
        ['4:4:0', [[1, 2], [1, 1], [1, 1]]],
        ['4:2:2', [[2, 1], [1, 1], [1, 1]]],
        ['4:2:0', [[2, 2], [1, 1], [1, 1]]],
        ['4:1:1', [[4, 1], [1, 1], [1, 1]]],
        ['4:1:0', [[4, 2], [1, 1], [1, 1]]],
        ['3:3:3', [[1, 1], [1, 1], [1, 1]]],
        ['3:3:0', [[1, 2], [1, 1], [1, 1]]],
        ['3:1:1', [[3, 1], [1, 1], [1, 1]]],
        ['3:1:0', [[3, 2], [1, 1], [1, 1]]],
    ])
    def test_samp_factor(self, Jab, samp_factor):
        self.logger.info(f'test_samp_factor({Jab})')
        # load image
        im = jpeglib.read_spatial('examples/IMG_0311.jpeg')
        # store with sampling factor
        im.samp_factor = Jab
        im.write_spatial(self.tmp.name)
        # load and check sampling factor
        jpeg = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(jpeg.samp_factor, samp_factor)


__all__ = ['TestNotation']
