
import logging
import numpy as np
import os
import pandas as pd
from parameterized import parameterized
from PIL import Image
import subprocess
import tempfile
import unittest

import jpeglib
from _defs import version_cluster

# https://www.sciencedirect.com/topics/computer-science/quantization-matrix
qt50_standard = np.array([
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]],
    [[17, 18, 24, 47, 99, 99, 99, 99],
     [18, 21, 26, 66, 99, 99, 99, 99],
     [24, 26, 56, 99, 99, 99, 99, 99],
     [47, 66, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99]],
    [[17, 18, 24, 47, 99, 99, 99, 99],
     [18, 21, 26, 66, 99, 99, 99, 99],
     [24, 26, 56, 99, 99, 99, 99, 99],
     [47, 66, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99]]
    ])


class TestDCT(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)

    def test_dct(self):
        """Test of lossless reading and writing of DCT."""
        self.logger.info("test_dct")
        # get original DCT coefficients
        jpeg = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # write and read again
        jpeg.write_dct(self.tmp.name)
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        # test DCT coefficients are the same after write/read
        np.testing.assert_array_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_array_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_array_equal(jpeg.Cr, jpeg2.Cr)
        np.testing.assert_array_equal(jpeg.qt, jpeg2.qt)

    def test_dct_grayscale(self):
        """Test of grayscale DCT coefficients."""
        self.logger.info("test_dct_grayscale")
        # get original DCT coefficients
        jpeg = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # write grayscale
        jpeglib.from_dct(
            Y=jpeg.Y,
            qt=jpeg.qt[:1],
        ).write_dct(self.tmp.name)
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        # test DCT coefficients are the same after write/read
        self.assertFalse(jpeg2.has_chrominance)
        np.testing.assert_array_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_array_equal(jpeg.qt[:1], jpeg2.qt)

    @parameterized.expand([
        [((1, 1), (1, 1), (1, 1))],
        [((2, 2), (2, 1), (2, 1))],
        [((2, 2), (1, 2), (1, 2))],
        [((3, 1), (1, 1), (1, 1))],
    ])
    def test_dct_samp_factor(self, samp_factor):
        """Test of reading DCT with different sampling factors."""
        self.logger.info("test_dct_samp_factor")
        # compress image with given sampling factor
        im = jpeglib.read_spatial("examples/IMG_0311.jpeg")
        im.samp_factor = samp_factor
        im.write_spatial(self.tmp.name)
        # read DCT coefficients
        jpeg = jpeglib.read_dct(self.tmp.name)
        ratio_Cb = [samp_factor[0][i]/samp_factor[1][i] for i in range(2)]
        ratio_Cr = [samp_factor[0][i]/samp_factor[2][i] for i in range(2)]
        # test DCT coefficients are the same after read
        self.assertEqual(jpeg.Y.shape[0], jpeg.Cb.shape[0]*ratio_Cb[1])
        self.assertEqual(jpeg.Y.shape[1], jpeg.Cb.shape[1]*ratio_Cb[0])
        self.assertEqual(jpeg.Y.shape[0], jpeg.Cr.shape[0]*ratio_Cr[1])
        self.assertEqual(jpeg.Y.shape[1], jpeg.Cr.shape[1]*ratio_Cr[0])
        # write DCT coefficients
        jpeglib.from_dct(
            Y=jpeg.Y,
            Cb=jpeg.Cb,
            Cr=jpeg.Cr,
            qt=jpeg.qt,
        ).write_dct(self.tmp.name)
        # test DCT coefficients are the same after write
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_array_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_array_equal(jpeg.Cr, jpeg2.Cr)

    def _compressed_dct(self, im, v):
        with jpeglib.version(v):
            im.write_spatial(self.tmp.name)
            return jpeglib.read_dct(self.tmp.name).load()

    @parameterized.expand([
        ['6b', '7', True],
        ['9d', '9e', True],
        ['mozjpeg201','mozjpeg300', False],
    ])
    def test_mismatch_baseline(self, v1, v2, equal_Y):
        """Compress with given two versions and observe differing output."""
        self.logger.info(f"test_mismatch_baseline_{v1}_{v2}")
        # compress image with given versions
        im = jpeglib.read_spatial("examples/IMG_0311.jpeg")
        Y1, (Cb1, Cr1), _ = self._compressed_dct(im, v1)
        Y2, (Cb2, Cr2), _ = self._compressed_dct(im, v2)
        # compare not equal
        if equal_Y:
            self.assertTrue((Y1 == Y2).all())
        else:
            self.assertFalse((Y1 == Y2).all())
        self.assertFalse((Cb1 == Cb2).all())
        self.assertFalse((Cr1 == Cr2).all())

    @parameterized.expand([
        ['6b', 'turbo120', 'turbo130', 'turbo140', 'turbo150', 'turbo200', 'turbo210'],
        ['7', '8', '8a', '8b', '8c', '8d', '9', '9a', '9b', '9c', '9d'],
        ['9e'],
        ['mozjpeg101', 'mozjpeg201'],
        ['mozjpeg300', 'mozjpeg403'],
    ], name_func=version_cluster)
    def test_equal_baseline(self, *versions):
        """Compress with given versions and observe the same output."""
        self.logger.info(f"test_mismatch_baseline_{'_'.join(versions)}")
        # compress image with reference
        im = jpeglib.read_spatial("examples/IMG_0311.jpeg")
        Y_ref, (Cb_ref, Cr_ref), qt_ref = self._compressed_dct(im, versions[0])
        # compress with each version and compare to the reference
        for v in versions:
            Y, (Cb, Cr), qt = self._compressed_dct(im, v)
            np.testing.assert_array_equal(Y_ref, Y)
            np.testing.assert_array_equal(Cb_ref, Cb)
            np.testing.assert_array_equal(Cr_ref, Cr)
            np.testing.assert_array_equal(qt_ref, qt)

    def test_dct_coefficient_decoder(self):
        """Test output against btlorch/dct-coefficient-decoder."""
        self.logger.info("test_dct_coefficient_decoder")
        # read DCT with jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # read DCT with dct-coefficient-decoder
        try:
            from decoder import PyCoefficientDecoder
        except (ModuleNotFoundError, ImportError) as e:  # error loading
            logging.error(
                f"invalid installation of dct-coefficient-decoder: {e}"
            )
            return
        d = PyCoefficientDecoder('examples/IMG_0311.jpeg')
        # convert to the same format
        qtT = np.stack([
            d.get_quantization_table(0),
            d.get_quantization_table(1),
            d.get_quantization_table(1),
        ])
        YT = (
            d.get_dct_coefficients(0)
            .reshape((im.height_in_blocks(0), im.width_in_blocks(0), 8, 8)))
        CbT = (
            d.get_dct_coefficients(1)
            .reshape((im.height_in_blocks(1), im.width_in_blocks(1), 8, 8)))
        CrT = (
            d.get_dct_coefficients(2)
            .reshape((im.height_in_blocks(2), im.width_in_blocks(2), 8, 8)))

        # test equal
        np.testing.assert_array_equal(im.Y, YT)
        np.testing.assert_array_equal(im.Cb, CbT)
        np.testing.assert_array_equal(im.Cr, CrT)
        np.testing.assert_array_equal(im.qt, qtT)

    def test_python_jpeg_toolbox(self):
        """Test output against daniellerch/python-jpeg-toolbox."""
        self.logger.info("test_python_jpeg_toolbox")
        # read DCT with jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # read DCT with python-jpeg-toolbox
        try:
            import jpeg_toolbox
        except (ModuleNotFoundError, ImportError) as e:
            logging.error(f"invalid installation of python-jpeg-toolbox: {e}")
            return
        img = jpeg_toolbox.load('examples/IMG_0311.jpeg')
        # convert to the same format
        YT = (
            img['coef_arrays'][0]
            .reshape((im.height_in_blocks(0), 8, -1, 8))
            .transpose((0, 2, 1, 3)))
        CbT = (
            img['coef_arrays'][1]
            .reshape((im.height_in_blocks(1), 8, -1, 8))
            .transpose((0, 2, 1, 3)))
        CrT = (
            img['coef_arrays'][2]
            .reshape((im.height_in_blocks(2), 8, -1, 8))
            .transpose((0, 2, 1, 3)))
        qtT = np.stack([
            img['quant_tables'][0],
            img['quant_tables'][1],
            img['quant_tables'][1],
        ])
        # test equal
        np.testing.assert_array_equal(im.qt, qtT)
        np.testing.assert_array_equal(im.Y, YT)
        np.testing.assert_array_equal(im.Cb, CbT)
        np.testing.assert_array_equal(im.Cr, CrT)

    def test_torchjpeg(self):
        """Test output against queuecumber/torchjpeg."""
        self.logger.info("test_torchjpeg")
        # read DCT with jpeglib
        jpeg = jpeglib.read_dct("examples/IMG_0791.jpeg")
        # read DCT with torchjpeg
        try:
            import torchjpeg.codec
            shape, qt, Y, CbCr = torchjpeg.codec.read_coefficients(  # noqa: E501
                "examples/IMG_0791.jpeg"
            )
        except ModuleNotFoundError as e:  # error loading
            logging.error(
                f"invalid installation of torchjpeg: {e}"
            )
            return

        # convert to the same format and test
        def get_full_shape(c):
            return np.array([
                np.prod(c.shape[i::2])
                for i in range(2)
            ])
        np.testing.assert_array_equal(
            get_full_shape(jpeg.Y),
            shape.numpy()[0]
        )
        np.testing.assert_array_equal(
            get_full_shape(jpeg.Cb),
            shape.numpy()[1]
        )
        np.testing.assert_array_equal(
            get_full_shape(jpeg.Cr),
            shape.numpy()[2]
        )
        np.testing.assert_array_equal(jpeg.qt, qt.numpy())
        np.testing.assert_array_equal(
            jpeg.Y,
            Y.numpy()[0]
        )  # noqa: E501
        np.testing.assert_array_equal(  # noqa: E501
            jpeg.Cb,
            CbCr[0].numpy()
        )
        np.testing.assert_array_equal(  # noqa: E501
            jpeg.Cr,
            CbCr[1].numpy()
        )

    def test_pil_qt(self):
        """Test the same QT is acquired from pillow."""
        self.logger.info("test_pil_qt")
        # read qt with jpeglib
        jpeg = jpeglib.read_dct("examples/IMG_0791.jpeg")
        # pillow
        im = Image.open("examples/IMG_0791.jpeg")
        # convert to the same format
        pil_qt = [
            (
                np.array(im.quantization[k])
                .reshape(8, 8)
            )
            for k in sorted(im.quantization)
        ]
        pil_qt = np.array([
            pil_qt[i]
            for i in jpeg.quant_tbl_no
        ])
        # test equal
        np.testing.assert_equal(pil_qt, jpeg.qt)
        im.close()

    def test_to_jpegio(self):
        """Test identical output of jpegio to DCTJPEGio interface."""
        self.logger.info("test_to_jpegio")
        # jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        im = jpeglib.to_jpegio(im)
        # jpegio
        try:
            import jpegio
        except (ModuleNotFoundError, ImportError) as e:
            logging.error(f"invalid installation of jpegio: {e}")
            return 1
        jpeg = jpegio.read('examples/IMG_0311.jpeg')
        # test quantization
        self.assertEqual(len(im.quant_tables), 2)
        np.testing.assert_array_equal(jpeg.quant_tables[0], im.quant_tables[0])
        np.testing.assert_array_equal(jpeg.quant_tables[1], im.quant_tables[1])
        # test DCT coefficients
        self.assertEqual(len(im.coef_arrays), 3)
        np.testing.assert_array_equal(jpeg.coef_arrays[0], im.coef_arrays[0])
        np.testing.assert_array_equal(jpeg.coef_arrays[1], im.coef_arrays[1])
        np.testing.assert_array_equal(jpeg.coef_arrays[2], im.coef_arrays[2])

    def test_jpegio_write(self):
        """Test writing of DCTJPEGio interface."""
        self.logger.info("test_jpegio_write")
        # jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        im = jpeglib.to_jpegio(im)
        # change quantization table in JPEGio
        im.qt[0, 0, 0] = 2
        self.assertEqual(im.qt[0, 0, 0], 2)
        # change quantization table in JPEG
        im.quant_tables[0][0, 0] = 1
        self.assertEqual(im.quant_tables[0][0, 0], 1)
        # write JPEGio and reload
        im.write(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        im2 = jpeglib.to_jpegio(im2)
        # check quantization table
        self.assertEqual(im.qt[0, 0, 0], 1)
        self.assertEqual(im.quant_tables[0][0, 0], 1)

    def test_dct_edit(self):
        """Test of changing DCT coefficient, e.g."""
        self.logger.info("test_dct_edit")
        # write with different qt
        jpeg = jpeglib.read_dct("examples/IMG_0311.jpeg")
        Yc = jpeg.Y.copy()
        jpeg.Y[0, 0, 0, 1] += 1  # change DCT (e.g., steganography)
        jpeg.write_dct(self.tmp.name)
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        # test matrix
        self.assertEqual(np.sum(jpeg2.Y - Yc), 1) # difference by 1
        self.assertEqual(
            tuple([int(i) for i in np.where(jpeg2.Y != Yc)]),
            (0, 0, 0, 1)
        )
        np.testing.assert_array_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_array_equal(jpeg.Cr, jpeg2.Cr)
        np.testing.assert_array_equal(jpeg.qt, jpeg2.qt)

    def test_qt1(self):
        """Test reading DCT of color JPEG with one QT."""
        self.logger.info("test_qt1")
        # read DCT of JPEG having one QT
        jpeg = jpeglib.read_dct("examples/qt1.jpeg")
        np.testing.assert_array_equal(jpeg.qt[0], jpeg.qt[1])
        np.testing.assert_array_equal(jpeg.qt[0], jpeg.qt[2])
        np.testing.assert_array_equal(jpeg.quant_tbl_no, np.array([0, 0, 0]))

    def test_from_dct(self):
        """Test creating synthetic JPEG DCT coefficients.

        Write them using from_dct call.
        """
        global qt50_standard
        self.logger.info("test_from_dct")
        # generate random DCT
        np.random.seed(12345)
        Y = np.random.randint(-127, 127, (32, 32, 8, 8), dtype=np.int16)
        Cb = np.random.randint(-127, 127, (16, 16, 8, 8), dtype=np.int16)
        Cr = np.random.randint(-127, 127, (16, 16, 8, 8), dtype=np.int16)
        # choose standard QT 50
        qt = qt50_standard
        # generating random QT causes wierd error messages,
        # increasing values in zigzag order are probably required

        # create synthetic JPEG using from_dct
        jpeglib.from_dct(
            Y=Y,
            Cb=Cb,
            Cr=Cr,
            qt=qt,
            quant_tbl_no=np.array([0,1,2]),
        ).write_dct(self.tmp.name)
        # load and compare
        jpeg = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(Y, jpeg.Y)
        np.testing.assert_array_equal(Cb, jpeg.Cb)
        np.testing.assert_array_equal(Cr, jpeg.Cr)
        np.testing.assert_array_equal(qt, jpeg.qt)



    # === tests with non-public software ===
    def test_rainer_MMSec(self):
        """Test output against Rainer's MMSec library."""
        self.logger.info("test_rainer_MMSec")

        # read image using jpeglib
        jpeg = jpeglib.read_dct('examples/IMG_0791.jpeg')
        jpeglib_dct = {
            'Y': jpeg.Y,
            'Cb': jpeg.Cb,
            'Cr': jpeg.Cr,
        }
        jpeglib_qt = {
            'Y': jpeg.qt[0],
            'Cb': jpeg.qt[1],
            'Cr': jpeg.qt[2],
        }
        try:
            # read image using Rainer's MMSec library
            for channel in ['Y', 'Cb', 'Cr']:

                # test DCT
                res = subprocess.call(
                        ["Rscript", "tests/test_rainer.R",  # script
                        "dct",  # mode - quantization table or DCT coefficients
                        channel,  # channel - Y, Cr, or Cb
                        "examples/IMG_0791.jpeg ",  # input file
                        self.tmp.name],  # output file
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                if res != 0:
                    raise Exception("Rainer's MMSec failed!")
                df = pd.read_csv(self.tmp.name, header=None)
                dctRainer = df.to_numpy()
                # get jpeglib output
                dct = jpeglib_dct[channel]
                # convert to same shape
                dctRainer = np.array(np.split(dctRainer, dct.shape[0], 0))
                dctRainer = dctRainer.reshape(*dctRainer.shape[:-1], 8, 8)
                # test equal
                np.testing.assert_equal(dctRainer, dct)

                # test QT
                res = subprocess.call(
                        "Rscript tests/test_rainer.R " + # script
                        "qt " + " " + # produce quantization table
                        channel + " " +# Y, Cr or Cb
                        "examples/IMG_0791.jpeg " + # input file
                        self.tmp.name, # output file

                    shell=True)
                if res != 0:
                    raise Exception("Rainer's script failed!")
                df = pd.read_csv("tmp/result.csv", header=None)
                qtRainer = df.to_numpy()
                # get jpeglib output
                qt = jpeglib_qt[channel]
                # compare quantization tables
                np.testing.assert_equal(qtRainer, qt)
        except Exception as e:
            self.logger.error(str(e))


__all__ = ["TestDCT"]
