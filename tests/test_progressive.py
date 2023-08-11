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

from _defs import LIBJPEG_VERSIONS, ALL_VERSIONS, VERSIONS_EXCLUDE_MOZ
import jpeglib


class TestProgressive(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    @parameterized.expand(ALL_VERSIONS)
    def test_read_progressive_flag(self, version):
        """Test if progressive_mode flag is correctly set."""
        jpeglib.version.set(version)
        self.logger.info(f"test_read_progressive_flag_{version}")

        # Test for progressive image
        if "mozjpeg" in str(version):
            im = jpeglib.read_spatial(f"tests/assets/images-{version}/testimg.jpg")
            self.assertTrue(im.progressive_mode), f"{version}"
        else:
            im = jpeglib.read_spatial(f"tests/assets/images-{version}/testimgp.jpg")
            self.assertTrue(im.progressive_mode), f"{version}"

        # Test for baseline sequential image
        if "mozjpeg" in str(version):
            im = jpeglib.read_spatial(f"tests/assets/images-{version}/testseq.jpg")
            self.assertFalse(im.progressive_mode)
        else:
            im = jpeglib.read_spatial(f"tests/assets/images-{version}/testimg.jpg")
            self.assertFalse(im.progressive_mode)

    @parameterized.expand(ALL_VERSIONS)
    def test_compressed_prog_vs_seq(self, version):
        """
        Test if the DCT coefficients of a progressive image are the same as for a sequential image.
        """
        jpeglib.version.set(version)
        uncompressed = "tests/assets/images-6b/testimg.ppm"
        self.logger.info(f"test_compressed_prog_vs_seq_{version}")

        # compress image progressivly
        rgb_uncompressed = np.array(Image.open(uncompressed))
        im = jpeglib.from_spatial(rgb_uncompressed)

        if "mozjpeg" in str(version):
            im = jpeglib.read_dct(f"tests/assets/images-{version}/testimg.jpg")
        else:
            im = jpeglib.read_dct(f"tests/assets/images-{version}/testimgp.jpg")

        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)

        np.testing.assert_array_equal(im.Y, im2.Y)
        np.testing.assert_array_equal(im.Cb, im2.Cb)
        np.testing.assert_array_equal(im.Cr, im2.Cr)
        np.testing.assert_array_equal(im.qt, im2.qt)

    @parameterized.expand(ALL_VERSIONS)
    def test_decompressed_prog_vs_seq(self, version):
        """
        Test if the pixel values of a progressive image are the same as for a sequential image.
        """

        jpeglib.version.set(version)
        self.logger.info(f"test_decompressed_prog_vs_seq_{version}")
        uncompressed = "tests/assets/images-6b/testimg.ppm"

        # Test jpeglib compressed images
        rgb_uncompressed = np.array(Image.open(uncompressed))
        im = jpeglib.from_spatial(rgb_uncompressed)

        im.write_spatial(self.tmp.name, flags=["+PROGRESSIVE_MODE"])
        spatial_progressive = jpeglib.read_spatial(self.tmp.name).spatial

        im.write_spatial(self.tmp.name, flags=["-PROGRESSIVE_MODE"])
        spatial_sequential = jpeglib.read_spatial(self.tmp.name).spatial

        np.testing.assert_array_equal(spatial_progressive, spatial_sequential)

        # TODO: Find out why images from cjpeg fail
        # # Test cjpeg compressed images
        # if "mozjpeg" in str(version):
        #     spatial_progressive = jpeglib.read_spatial(
        #         f"tests/assets/images-{version}/testimg.jpg"
        #     ).spatial
        #     spatial_sequential = jpeglib.read_spatial(
        #         f"tests/assets/images-{version}/testseq.jpg"
        #     ).spatial
        # else:
        #     spatial_progressive = jpeglib.read_spatial(
        #         f"tests/assets/images-{version}/testimgp.jpg"
        #     ).spatial
        #     spatial_sequential = jpeglib.read_spatial(
        #         f"tests/assets/images-{version}/testimg.jpg"
        #     ).spatial
        # np.testing.assert_array_equal(spatial_progressive, spatial_sequential)

    @parameterized.expand(VERSIONS_EXCLUDE_MOZ)
    def test_progressive_standard_scanscript_color(self, version):
        """Test standard script is used when compressing with libjpeg or libjpeg turbo."""
        jpeglib.version.set(version)
        self.logger.info(f"test_progressive_standard_scanscript_color_{version}")
        # compress as progressive
        rgb = np.random.randint(0, 256, (64, 64, 3), dtype="uint8")
        im = jpeglib.from_spatial(rgb)
        im.write_spatial(self.tmp.name, flags=["+PROGRESSIVE_MODE"])
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

    @parameterized.expand(VERSIONS_EXCLUDE_MOZ)
    def test_progressive_standard_scanscript_grayscale(self, version):
        """Test standard script is used when compressing with libjpeg or libjpeg turbo."""
        jpeglib.version.set(version)
        self.logger.info(f"test_progressive_standard_scanscript_grayscale_{version}")
        # compress as progressive
        rgb = np.random.randint(0, 256, (64, 64, 1), dtype="uint8")
        im = jpeglib.from_spatial(rgb)
        im.write_spatial(self.tmp.name, flags=["+PROGRESSIVE_MODE"])
        # read buffered
        im = jpeglib.read_spatial(self.tmp.name, buffered=True)
        # scan 1
        np.testing.assert_array_equal(im.scans[0].components, [0])
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
        np.testing.assert_array_equal(im.scans[2].components, [0])
        np.testing.assert_array_equal(im.scans[2].Ss, 6)
        np.testing.assert_array_equal(im.scans[2].Se, 63)
        np.testing.assert_array_equal(im.scans[2].Ah, 0)
        np.testing.assert_array_equal(im.scans[2].Al, 2)
        # scan 4
        np.testing.assert_array_equal(im.scans[3].components, [0])
        np.testing.assert_array_equal(im.scans[3].Ss, 1)
        np.testing.assert_array_equal(im.scans[3].Se, 63)
        np.testing.assert_array_equal(im.scans[3].Ah, 2)
        np.testing.assert_array_equal(im.scans[3].Al, 1)
        # scan 5
        np.testing.assert_array_equal(im.scans[4].components, [0])
        np.testing.assert_array_equal(im.scans[4].Ss, 0)
        np.testing.assert_array_equal(im.scans[4].Se, 0)
        np.testing.assert_array_equal(im.scans[4].Ah, 1)
        np.testing.assert_array_equal(im.scans[4].Al, 0)
        # scan 6
        np.testing.assert_array_equal(im.scans[5].components, [0])
        np.testing.assert_array_equal(im.scans[5].Ss, 1)
        np.testing.assert_array_equal(im.scans[5].Se, 63)
        np.testing.assert_array_equal(im.scans[5].Ah, 1)
        np.testing.assert_array_equal(im.scans[5].Al, 0)

    def test_progressive_trellis_djpeg(self):  # TODO: Nora
        """Test progressive with Trellis is identical as djpeg output."""
        self.logger.info("test_progressive_trellis_djpeg")

        im = jpeglib.read_spatial(
            "tests/assets/progressive_trellis.jpeg", buffered=True
        )
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

    @parameterized.expand(VERSIONS_EXCLUDE_MOZ)
    def test_sequential_scan_script_color(self, version):
        """
        Test that a sequential image has a correct scan script.
        """
        jpeglib.version.set(version)
        self.logger.info(f"test_sequential_scan_script_{version}")
        scans = [
            jpeglib.Scan(
                components=np.array([0, 1, 2]),
                Ss=0,
                Se=63,
                Ah=0,
                Al=0,
                dc_tbl_no=np.array([0, 1, 1]),
                ac_tbl_no=np.array([0, 1, 1]),
            )
        ]
        rgb = np.array(Image.open("tests/assets/00001.tif"))
        im = jpeglib.from_spatial(rgb)
        im.write_spatial(self.tmp.name)
        im2 = jpeglib.read_spatial(self.tmp.name, buffered=True)
        for i, scan in enumerate(scans):
            np.testing.assert_array_equal(im2.scans[i].components, scan.components)
            np.testing.assert_array_equal(im2.scans[i].Ss, scan.Ss)
            np.testing.assert_array_equal(im2.scans[i].Se, scan.Se)
            np.testing.assert_array_equal(im2.scans[i].Ah, scan.Ah)
            np.testing.assert_array_equal(im2.scans[i].Al, scan.Al)

    @parameterized.expand(VERSIONS_EXCLUDE_MOZ)
    def test_sequential_scan_script_grayscale(self, version):
        """
        Test that a sequential image has a correct scan script.
        """
        jpeglib.version.set(version)
        self.logger.info(f"test_sequential_scan_script_grayscale_{version}")
        scans = [
            jpeglib.Scan(
                components=np.array([0]),
                Ss=0,
                Se=63,
                Ah=0,
                Al=0,
                dc_tbl_no=np.array([0]),
                ac_tbl_no=np.array([0]),
            )
        ]
        rgb = np.random.randint(0, 256, (64, 64, 1), dtype="uint8")
        im = jpeglib.from_spatial(rgb)
        im.write_spatial(self.tmp.name)
        im2 = jpeglib.read_spatial(self.tmp.name, buffered=True)
        for i, scan in enumerate(scans):
            np.testing.assert_array_equal(im2.scans[i].components, scan.components)
            np.testing.assert_array_equal(im2.scans[i].Ss, scan.Ss)
            np.testing.assert_array_equal(im2.scans[i].Se, scan.Se)
            np.testing.assert_array_equal(im2.scans[i].Ah, scan.Ah)
            np.testing.assert_array_equal(im2.scans[i].Al, scan.Al)

    def test_progressive_set_custom_script(self):
        """Test setting custom script for progressive JPEG compression."""
        self.logger.info("test_progressive_set_progressive_script")
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
        rgb = np.array(Image.open("tests/assets/00001.tif"))
        im = jpeglib.from_spatial(rgb, scans=scans)
        im.write_spatial(self.tmp.name)
        im2 = jpeglib.read_spatial(self.tmp.name, buffered=True)
        for i, scan in enumerate(scans):
            np.testing.assert_array_equal(im2.scans[i].components, scan.components)
            np.testing.assert_array_equal(im2.scans[i].Ss, scan.Ss)
            np.testing.assert_array_equal(im2.scans[i].Se, scan.Se)
            np.testing.assert_array_equal(im2.scans[i].Ah, scan.Ah)
            np.testing.assert_array_equal(im2.scans[i].Al, scan.Al)

    # TODO: test mozjpeg uses optimized scan script? - How?
    # TODO: tests for social network scan scripts?
    # TODO: ?


__all__ = ["TestProgressive"]
