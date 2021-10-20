
import logging
import numpy as np
import os
import shutil
import sys
import unittest
import warnings

sys.path.append('.')
import jpeglib

class TestDCT(unittest.TestCase):
    logger = logging.getLogger(__name__)
    def setUp(self):
        try: shutil.rmtree("tmp")
        except: pass
        finally: os.mkdir("tmp")
    def tearDown(self):
        shutil.rmtree("tmp")

    def test_dct_coefficient_decoder(self):
        # jpeglib
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()

        # dct-coefficient-decoder
        try:
            from decoder import PyCoefficientDecoder 
        except Exception as e: # error loading
            logging.info(f"invalid installation of dct-coefficient-decoder: {e}")
            return
        d = PyCoefficientDecoder('examples/IMG_0791.jpeg')
        # process
        qtT = np.stack([d.get_quantization_table(i) for i in range(3)])
        YT = d.get_dct_coefficients(0).reshape((1,int(d.image_width/8),-1,8,8), order='F')
        CbCrT = np.stack([
            d.get_dct_coefficients(1).reshape((int(d.image_width/8/2),-1,8,8), order='F'),
            d.get_dct_coefficients(2).reshape((int(d.image_width/8/2),-1,8,8), order='F'),
        ])

        # test DCT coefficients
        np.testing.assert_array_equal(Y, YT)
        np.testing.assert_array_equal(CbCr, CbCrT)
        # test quantization
        np.testing.assert_array_equal(qt, qtT)
        
    def test_python_jpeg_toolbox(self):
        # jpeglib
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()
        print(Y.shape)
        # jpeg-toolbox
        try:
            import jpeg_toolbox
        except Exception as e:
            logging.info(f"invalid installation of python-jpeg-toolbox: {e}")
            return
        img = jpeg_toolbox.load('examples/IMG_0791.jpeg')
        # process
        YT = img['coef_arrays'][0].reshape((1,int(img['image_width']/8),-1,8,8))
        CbCrT = np.stack([
            img['coef_arrays'][1].reshape((int(img['image_width']/8/2),-1,8,8), order='F'),
            img['coef_arrays'][2].reshape((int(img['image_width']/8/2),-1,8,8), order='F')
        ])
        qtT = np.concatenate([img['quant_tables'], img['quant_tables'][1:]])

        print(Y.shape, CbCr.shape, qt.shape)
        print(YT.shape, CbCrT.shape, qtT.shape)

        # test DCT coefficients
        np.testing.assert_array_equal(Y, YT)
        np.testing.assert_array_equal(CbCr, CbCrT)
        # test quantization
        np.testing.assert_array_equal(qt, qtT)
    # def test_torchjpeg(self):
    #     # read image
    #     im = jpeglib.JPEG("examples/IMG_0791.jpeg")
    #     Y,CbCr,qt = im.read_dct()
    #     # read by torchjpeg
    #     shapeT,qtT,YT,CbCrT = torchjpeg.codec.read_coefficients("examples/IMG_0791.jpeg")

    #     # compare
    #     np.testing.assert_array_equal(im.shape, shapeT.numpy()[0,::-1])
    #     np.testing.assert_array_equal(im.shape, shapeT.numpy()[1,::-1] * 2)
    #     np.testing.assert_array_equal(im.shape, shapeT.numpy()[2,::-1] * 2)
    #     np.testing.assert_array_equal(qt, qtT.numpy())
    #     np.testing.assert_array_equal(Y, np.einsum('abcde->acbed', YT.numpy()))
    #     np.testing.assert_array_equal(CbCr[0], np.einsum('bcde->cbed', CbCrT[0].numpy()))
    #     np.testing.assert_array_equal(CbCr[1], np.einsum('bcde->cbed', CbCrT[1].numpy()))

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
    #         self._rainer_qt('Cr', qt[2])

__all__ = ["TestDCT"]