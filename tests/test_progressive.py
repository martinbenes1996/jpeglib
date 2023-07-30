"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import tempfile
import unittest

from _defs import LIBJPEG_VERSIONS
import jpeglib


class TestProgressive(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    def test_read_progressive_flag(self):
        self.logger.info("test_read_progressive_flag")

        im = jpeglib.read_spatial("examples/images-6b/testprog.jpg")
        self.assertTrue(im.progressive_mode)

        im = jpeglib.read_spatial("examples/images-6b/testimg.jpg")
        self.assertFalse(im.progressive_mode)

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_progressive_decompress_vs_pil(self, version):
        """Test on test images from libjpeg."""
        self.logger.info(f"test_progressive_decompress_vs_pil_{version}")
        jpeglib.version.set(version)

        # im_prog = jpeglib.read_spatial(
        _ = jpeglib.read_spatial(
            f'examples/images-{version}/testprog.jpg',
            flags=[
                '+PROGRESSIVE_MODE',
                '+DO_FANCY_UPSAMPLING',
                '+DO_BLOCK_SMOOTHING'
            ]
        )
        # im_prog = jpeglib.read_spatial(
        _ = np.array(Image.open(
            f'examples/images-{version}/testprog.jpg'
        ))
        # np.testing.assert_array_almost_equal(im_prog.spatial, rgb_pil) # TODO: Nora

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_progressive_dct(self, version):
        self.logger.info(f"test_progressive_dct_{version}")
        # load dct - to fix
        im = jpeglib.read_dct(f'examples/images-{version}/testprog.jpg')
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(im.Y, im2.Y)
        # D = np.abs((CbCr.astype(np.int) - CbCr2.astype(np.int)))
        # print((D != 0).sum()) # 1073 mismatches
        # np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(im.qt, im2.qt)

    @parameterized.expand(LIBJPEG_VERSIONS)
    def test_progressive_sequential(self, version):
        self.logger.info(f"test_progressive_sequential_{version}")

        # load progressive image
        im_seq = jpeglib.read_spatial(
            f'examples/images-{version}/testimg.jpg',
            flags=['-PROGRESSIVE_MODE']
        )
        im_prog = jpeglib.read_spatial(
            f'examples/images-{version}/testimgp.jpg',
            flags=['+PROGRESSIVE_MODE']
        )
        np.testing.assert_array_almost_equal(im_seq.spatial, im_prog.spatial)

    def test_progressive_standard_scanscript(self):
        self.logger.info('test_progressive_standard_scanscript')
        jpeglib.version.set('9e')
        # compress as progressive
        rgb = np.random.randint(0, 256, (64, 64, 3), dtype='uint8')
        im = jpeglib.from_spatial(rgb)
        im.write_spatial(self.tmp.name, flags=['+PROGRESSIVE_MODE'])
        # read buffered
        im = jpeglib.read_spatial(self.tmp.name, buffered=True)
        # scan 1
        np.testing.assert_array_equal(im.scans[0].components, [0, 1, 2])
        np.testing.assert_array_equal(im.scans[0].Ss, 0)
        np.testing.assert_array_equal(im.scans[0].Se, 0)
        np.testing.assert_array_equal(im.scans[0].Ah, 0)
        np.testing.assert_array_equal(im.scans[0].Al, 1)
        # scan 2
        np.testing.assert_array_equal(im.scans[1].components, [0])
        np.testing.assert_array_equal(im.scans[1].Ss, 1)
        np.testing.assert_array_equal(im.scans[1].Se, 5)
        np.testing.assert_array_equal(im.scans[1].Ah, 0)
        np.testing.assert_array_equal(im.scans[1].Al, 2)
        # scan 3
        np.testing.assert_array_equal(im.scans[2].components, [2])
        np.testing.assert_array_equal(im.scans[2].Ss, 1)
        np.testing.assert_array_equal(im.scans[2].Se, 63)
        np.testing.assert_array_equal(im.scans[2].Ah, 0)
        np.testing.assert_array_equal(im.scans[2].Al, 1)
        # scan 4
        np.testing.assert_array_equal(im.scans[3].components, [1])
        np.testing.assert_array_equal(im.scans[3].Ss, 1)
        np.testing.assert_array_equal(im.scans[3].Se, 63)
        np.testing.assert_array_equal(im.scans[3].Ah, 0)
        np.testing.assert_array_equal(im.scans[3].Al, 1)
        # scan 5
        np.testing.assert_array_equal(im.scans[4].components, [0])
        np.testing.assert_array_equal(im.scans[4].Ss, 6)
        np.testing.assert_array_equal(im.scans[4].Se, 63)
        np.testing.assert_array_equal(im.scans[4].Ah, 0)
        np.testing.assert_array_equal(im.scans[4].Al, 2)
        # scan 6
        np.testing.assert_array_equal(im.scans[5].components, [0])
        np.testing.assert_array_equal(im.scans[5].Ss, 1)
        np.testing.assert_array_equal(im.scans[5].Se, 63)
        np.testing.assert_array_equal(im.scans[5].Ah, 2)
        np.testing.assert_array_equal(im.scans[5].Al, 1)
        # scan 7
        np.testing.assert_array_equal(im.scans[6].components, [0, 1, 2])
        np.testing.assert_array_equal(im.scans[6].Ss, 0)
        np.testing.assert_array_equal(im.scans[6].Se, 0)
        np.testing.assert_array_equal(im.scans[6].Ah, 1)
        np.testing.assert_array_equal(im.scans[6].Al, 0)
        # scan 8
        np.testing.assert_array_equal(im.scans[7].components, [2])
        np.testing.assert_array_equal(im.scans[7].Ss, 1)
        np.testing.assert_array_equal(im.scans[7].Se, 63)
        np.testing.assert_array_equal(im.scans[7].Ah, 1)
        np.testing.assert_array_equal(im.scans[7].Al, 0)
        # scan 9
        np.testing.assert_array_equal(im.scans[8].components, [1])
        np.testing.assert_array_equal(im.scans[8].Ss, 1)
        np.testing.assert_array_equal(im.scans[8].Se, 63)
        np.testing.assert_array_equal(im.scans[8].Ah, 1)
        np.testing.assert_array_equal(im.scans[8].Al, 0)
        # scan 10
        np.testing.assert_array_equal(im.scans[9].components, [0])
        np.testing.assert_array_equal(im.scans[9].Ss, 1)
        np.testing.assert_array_equal(im.scans[9].Se, 63)
        np.testing.assert_array_equal(im.scans[9].Ah, 1)
        np.testing.assert_array_equal(im.scans[9].Al, 0)

    def test_progressive_trellis_djpeg(self):
        """Test progressive with Trellis is identical as djpeg output."""
        self.logger.info('test_progressive_trellis_djpeg')

        im = jpeglib.read_spatial('examples/progressive_trellis.jpeg', buffered=True)
        # scan 1
        np.testing.assert_array_equal(im.scans[0].components, [0, 1, 2])
        np.testing.assert_array_equal(im.scans[0].Ss, 0)
        np.testing.assert_array_equal(im.scans[0].Se, 0)
        np.testing.assert_array_equal(im.scans[0].Ah, 0)
        np.testing.assert_array_equal(im.scans[0].Al, 0)
        # scan 2
        np.testing.assert_array_equal(im.scans[1].components, [0])
        np.testing.assert_array_equal(im.scans[1].Ss, 1)
        np.testing.assert_array_equal(im.scans[1].Se, 2)
        np.testing.assert_array_equal(im.scans[1].Ah, 0)
        np.testing.assert_array_equal(im.scans[1].Al, 1)
        # scan 3
        np.testing.assert_array_equal(im.scans[2].components, [0])
        np.testing.assert_array_equal(im.scans[2].Ss, 3)
        np.testing.assert_array_equal(im.scans[2].Se, 63)
        np.testing.assert_array_equal(im.scans[2].Ah, 0)
        np.testing.assert_array_equal(im.scans[2].Al, 1)
        # scan 4
        np.testing.assert_array_equal(im.scans[3].components, [0])
        np.testing.assert_array_equal(im.scans[3].Ss, 1)
        np.testing.assert_array_equal(im.scans[3].Se, 63)
        np.testing.assert_array_equal(im.scans[3].Ah, 1)
        np.testing.assert_array_equal(im.scans[3].Al, 0)
        # scan 5
        np.testing.assert_array_equal(im.scans[4].components, [1])
        np.testing.assert_array_equal(im.scans[4].Ss, 1)
        np.testing.assert_array_equal(im.scans[4].Se, 2)
        np.testing.assert_array_equal(im.scans[4].Ah, 0)
        np.testing.assert_array_equal(im.scans[4].Al, 0)
        # scan 6
        np.testing.assert_array_equal(im.scans[5].components, [1])
        np.testing.assert_array_equal(im.scans[5].Ss, 3)
        np.testing.assert_array_equal(im.scans[5].Se, 63)
        np.testing.assert_array_equal(im.scans[5].Ah, 0)
        np.testing.assert_array_equal(im.scans[5].Al, 0)
        # scan 7
        np.testing.assert_array_equal(im.scans[6].components, [2])
        np.testing.assert_array_equal(im.scans[6].Ss, 1)
        np.testing.assert_array_equal(im.scans[6].Se, 2)
        np.testing.assert_array_equal(im.scans[6].Ah, 0)
        np.testing.assert_array_equal(im.scans[6].Al, 0)
        # scan 8
        np.testing.assert_array_equal(im.scans[7].components, [2])
        np.testing.assert_array_equal(im.scans[7].Ss, 3)
        np.testing.assert_array_equal(im.scans[7].Se, 63)
        np.testing.assert_array_equal(im.scans[7].Ah, 0)
        np.testing.assert_array_equal(im.scans[7].Al, 0)

    def test_progressive_set_scanscript(self):
        self.logger.info('test_progressive_set_scanscript')
        scans = [
            jpeglib.Scan(
                components=np.array([0, 1, 2]),
                Ss=0,
                Se=0,
                Ah=0,
                Al=2,
                dc_tbl_no=np.array([0, 1, 1]),
                ac_tbl_no=np.array([0, 1, 1]),
            ),  # DC w/o 2LSB
            jpeglib.Scan(
                components=np.array([0, 1, 2]),
                Ss=0,
                Se=0,
                Ah=2,
                Al=1,
                dc_tbl_no=np.array([0, 1, 1]),
                ac_tbl_no=np.array([0, 1, 1]),
            ),  # DC 2nd LSB
            jpeglib.Scan(
                components=np.array([0, 1, 2]),
                Ss=0,
                Se=0,
                Ah=1,
                Al=0,
                dc_tbl_no=np.array([0, 1, 1]),
                ac_tbl_no=np.array([0, 1, 1]),
            ),  # DC LSB
            jpeglib.Scan(
                components=np.array([1]),
                Ss=1,
                Se=63,
                Ah=0,
                Al=0,
                dc_tbl_no=np.array([1]),
                ac_tbl_no=np.array([1]),
            ),  # AC of Cb
            jpeglib.Scan(
                components=np.array([2]),
                Ss=1,
                Se=63,
                Ah=0,
                Al=0,
                dc_tbl_no=np.array([1]),
                ac_tbl_no=np.array([1]),
            ),  # AC of Cr
            jpeglib.Scan(
                components=np.array([0]),
                Ss=1,
                Se=63,
                Ah=0,
                Al=0,
                dc_tbl_no=np.array([0]),
                ac_tbl_no=np.array([0]),
            ),  # AC of Y
        ]
        rgb = np.array(Image.open('examples/00001.tif'))
        im = jpeglib.from_spatial(rgb, scans=scans)
        im.write_spatial(self.tmp.name)
        #
        im2 = jpeglib.read_spatial(self.tmp.name, buffered=True)
        for i, scan in enumerate(scans):
            np.testing.assert_array_equal(
                im2.scans[i].components, scan.components)
            np.testing.assert_array_equal(im2.scans[i].Ss, scan.Ss)
            np.testing.assert_array_equal(im2.scans[i].Se, scan.Se)
            np.testing.assert_array_equal(im2.scans[i].Ah, scan.Ah)
            np.testing.assert_array_equal(im2.scans[i].Al, scan.Al)

    # TODO: tests for social network scan scripts?
    # TODO: ?


__all__ = ["TestProgressive"]
