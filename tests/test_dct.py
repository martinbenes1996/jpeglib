
import logging
import numpy as np
import tempfile
import unittest

import jpeglib

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
        self.tmp = tempfile.NamedTemporaryFile(suffix='jpeg')

    def tearDown(self):
        del self.tmp

    def test_synthetic_dct(self):
        """"""
        global qt50_standard
        self.logger.info("test_synthetic_dct")
        # create synthetic JPEG
        np.random.seed(12345)
        Y = (np.random.rand(32, 32, 8, 8)*255-128).astype(np.int16)
        Cb = (np.random.rand(16, 16, 8, 8)*255-128).astype(np.int16)
        Cr = (np.random.rand(16, 16, 8, 8)*255-128).astype(np.int16)
        # qt = (np.random.rand(2, 8, 8)*8+1).astype(np.int16)
        qt = qt50_standard
        jpeglib.from_dct(
            Y=Y,
            Cb=Cb,
            Cr=Cr,
            qt=qt,
            quant_tbl_no=np.array([0,1,2]),
        ).write_dct(self.tmp.name)
        # load and compare
        jpeg = jpeglib.read_dct(self.tmp.name)
        # print(jpeg.qt)
        np.testing.assert_array_equal(Y, jpeg.Y)
        np.testing.assert_array_equal(Cb, jpeg.Cb)
        np.testing.assert_array_equal(Cr, jpeg.Cr)
        np.testing.assert_array_equal(qt, jpeg.qt)

    def test_dct_coefficient_decoder(self):
        self.logger.info("test_dct_coefficient_decoder")
        # jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # dct-coefficient-decoder
        try:
            from decoder import PyCoefficientDecoder
        except (ModuleNotFoundError,ImportError) as e:  # error loading
            logging.error(
                f"invalid installation of dct-coefficient-decoder: {e}"
            )
            return
        d = PyCoefficientDecoder('examples/IMG_0311.jpeg')
        # process
        qtT = np.stack([
            d.get_quantization_table(0),
            d.get_quantization_table(1),
            d.get_quantization_table(1),
        ])
        YT = (
            d.get_dct_coefficients(0)
            .reshape((im.width_in_blocks(0), -1, 8, 8), order='F')
            .transpose((1, 0, 2, 3)))
        CbT = (
            d.get_dct_coefficients(1)
            .reshape((im.width_in_blocks(1), -1, 8, 8), order='F')
            .transpose((1, 0, 2, 3)))
        CrT = (
            d.get_dct_coefficients(2)
            .reshape((im.width_in_blocks(2), -1, 8, 8), order='F')
            .transpose((1, 0, 2, 3)))
        # test DCT coefficients
        np.testing.assert_array_equal(im.Y, YT)
        np.testing.assert_array_equal(im.Cb, CbT)
        np.testing.assert_array_equal(im.Cr, CrT)
        # test quantization
        np.testing.assert_array_equal(im.qt, qtT)

    def test_python_jpeg_toolbox(self):
        self.logger.info("test_python_jpeg_toolbox")
        # jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        # jpeg-toolbox
        try:
            import jpeg_toolbox
        except (ModuleNotFoundError,ImportError) as e:
            logging.error(f"invalid installation of python-jpeg-toolbox: {e}")
            return
        img = jpeg_toolbox.load('examples/IMG_0311.jpeg')
        # process
        YT = (
            img['coef_arrays'][0]
            .reshape((im.height_in_blocks(0), 8, -1, 8))
            .transpose((0, 2, 3, 1)))
        CbT = (
            img['coef_arrays'][1]
            .reshape((im.height_in_blocks(1), 8, -1, 8))
            .transpose((0, 2, 3, 1)))
        CrT = (
            img['coef_arrays'][2]
            .reshape((im.height_in_blocks(2), 8, -1, 8))
            .transpose((0, 2, 3, 1)))
        qtT = np.stack([
            img['quant_tables'][0],
            img['quant_tables'][1],
            img['quant_tables'][1],
        ])

        # test quantization
        np.testing.assert_array_equal(im.qt, qtT)
        # test DCT coefficients
        np.testing.assert_array_equal(im.Y, YT)
        np.testing.assert_array_equal(im.Cb, CbT)
        np.testing.assert_array_equal(im.Cr, CrT)

    def test_to_jpegio(self):
        self.logger.info("test_to_jpegio")
        # jpeglib
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        im = jpeglib.to_jpegio(im)
        # jpegio
        try:
            import jpegio
        except (ModuleNotFoundError,ImportError) as e:
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

    def test_dct(self):
        self.logger.info("test_dct")
        # pass qt through
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        # test matrix
        np.testing.assert_array_equal(im.Y, im2.Y)
        np.testing.assert_array_equal(im.Cb, im2.Cb)
        np.testing.assert_array_equal(im.Cr, im2.Cr)
        np.testing.assert_array_equal(im.qt, im2.qt)

    def test_dct_qt(self):
        self.logger.info("test_dct_qt")
        # pass qt through
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        im.qt = im.qt
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        # test matrix
        np.testing.assert_array_equal(im.Y, im2.Y)
        np.testing.assert_array_equal(im.Cb, im2.Cb)
        np.testing.assert_array_equal(im.Cr, im2.Cr)
        np.testing.assert_array_equal(im.qt, im2.qt)

    def test_dct_qt50(self):
        self.logger.info("test_dct_qt50")
        global qt50_standard
        # pass qt through
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        im.qt = qt50_standard
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        # test matrix
        np.testing.assert_array_equal(im2.qt, qt50_standard)

    def test_dct_qt_edit(self):
        self.logger.info("test_dct_qt_edit")
        # write with different qt
        im = jpeglib.read_dct("examples/IMG_0311.jpeg")
        qt = (im.qt).copy()
        qt[0, 4, 4] = 1  # change qt
        im.qt = qt
        im.write_dct(self.tmp.name)
        im2 = jpeglib.read_dct(self.tmp.name)
        # test matrix
        np.testing.assert_array_equal(im.Y, im2.Y)
        np.testing.assert_array_equal(im.Cb, im2.Cb)
        np.testing.assert_array_equal(im.Cr, im2.Cr)
        np.testing.assert_array_equal(im.qt, im2.qt)

    def test_qt1(self):
        self.logger.info("test_qt1")
        im = jpeglib.read_dct("examples/qt1.jpeg")
        np.testing.assert_array_equal(im.qt[0], im.qt[1])

    def test_from_dct(self):
        self.logger.info("test_from_dct")
        # generate random DCT
        Y = np.random.randint(-127, 127, (2, 2, 8, 8), dtype=np.int16)
        Cb = np.random.randint(-127, 127, (1, 1, 8, 8), dtype=np.int16)
        Cr = np.random.randint(-127, 127, (1, 1, 8, 8), dtype=np.int16)
        jpeg = jpeglib.from_dct(Y, Cb, Cr)
        jpeg.write_dct(self.tmp.name)
        # reopen and compare
        jpeg2 = jpeglib.read_dct(self.tmp.name)
        np.testing.assert_array_equal(jpeg.Y, jpeg2.Y)
        np.testing.assert_array_equal(jpeg.Cb, jpeg2.Cb)
        np.testing.assert_array_equal(jpeg.Cr, jpeg2.Cr)

    def test_torchjpeg(self):
        self.logger.info("test_torchjpeg")
        # read image
        jpeg = jpeglib.read_dct("examples/IMG_0791.jpeg")
        # read by torchjpeg
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

        # compare
        get_full_shape = lambda c: (
            np.array([np.prod(c.shape[i::2]) for i in range(2)])
        )
        np.testing.assert_array_equal(
            get_full_shape(jpeg.Y),
            shape.numpy()[0,::]
        )
        np.testing.assert_array_equal(
            get_full_shape(jpeg.Cb),
            shape.numpy()[1,::]
        )
        np.testing.assert_array_equal(
            get_full_shape(jpeg.Cr),
            shape.numpy()[2,::]
        )
        np.testing.assert_array_equal(jpeg.qt, qt.numpy())
        np.testing.assert_array_equal(
            jpeg.Y,
            np.einsum('abcde->bced', Y.numpy())
        )  # noqa: E501
        np.testing.assert_array_equal(  # noqa: E501
            jpeg.Cb,
            np.einsum('bcde->bced', CbCr[0].numpy())
        )
        np.testing.assert_array_equal(  # noqa: E501
            jpeg.Cr,
            np.einsum('bcde->bced', CbCr[1].numpy())
        )

    # def _rainer_qt(self, outchannel, qt):
    #     # test quantization tables
    #     # get Rainer's (reference) result
    #     res = subprocess.call(
    #         "Rscript tests/test_rainer.R " + # script
    #         "qt " + " " + # produce quantization table
    #         outchannel + " " +# Y, Cr or Cb
    #         "examples/IMG_0791.jpeg " + # input file
    #         "tmp/result.csv", # output file
    #         shell=True)
    #     if res != 0:
    #         raise Exception("Rainer's script failed!")
    #     df = pd.read_csv("tmp/result.csv", header=None)
    #     qtRainer = df.to_numpy()

    #     # compare quantization tables
    #     np.testing.assert_equal(qtRainer, qt)

    # def _rainer_dct(self, outchannel, dct):
    #     # test DCT coefficients
    #     # get Rainer's (reference) result
    #     res = subprocess.call(
    #         "Rscript tests/test_rainer.R " + # script
    #         "dct " + " " + # produce quantization table
    #         outchannel + " " +# Y, Cr or Cb
    #         "examples/IMG_0791.jpeg " + # input file
    #         "tmp/result.csv", # output file
    #         shell=True)
    #     if res != 0:
    #         raise Exception("Rainer's script failed!")
    #     df = pd.read_csv("tmp/result.csv", header=None)
    #     dctRainer = df.to_numpy()
    #     # change shape
    #     dctRainer = np.array(np.split(dctRainer, dct.shape[0], 0))
    #     dctRainer = dctRainer.reshape(*dctRainer.shape[:-1], 8, 8)

    #     # compare DCT coefficient matrices
    #     np.testing.assert_equal(dctRainer, dct)

    # def test_rainer_dct_Y(self):
    #     # read images
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         Y,_,_ = im.read_dct()
    #         # call test
    #         self._rainer_dct('Y', Y[0])

    # def test_rainer_dct_Cb(self):
    #     # read image
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         _,CbCr,_ = im.read_dct()
    #         # call test
    #         self._rainer_dct('Cb', CbCr[0])

    # def test_rainer_dct_Cr(self):
    #     # read image
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         _,CbCr,_ = im.read_dct()
    #         # call test
    #         self._rainer_dct('Cr', CbCr[1])

    # def test_rainer_qt_Y(self):
    #     # read image
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         _,_,qt = im.read_dct()
    #         # call test
    #         self._rainer_qt('Y', qt[0])

    # def test_rainer_qt_Cb(self):
    #     # read image
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         _,_,qt = im.read_dct()
    #         # call test
    #         self._rainer_qt('Cb', qt[1])

    # def test_rainer_qt_Cr(self):
    #     # read image
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         _,_,qt = im.read_dct()
    #         # call test
    #         self._rainer_qt('Cr', qt[1])


__all__ = ["TestDCT"]
