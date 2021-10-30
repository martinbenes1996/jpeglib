
import logging
import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import sys
import unittest
import warnings

# try: import torchjpeg.codec
# except: pass

sys.path.append('.')
import jpeglib

# https://www.sciencedirect.com/topics/computer-science/quantization-matrix
qt50_standard = np.array([
    [[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]],
    [[17,18,24,47,99,99,99,99],
     [18,21,26,66,99,99,99,99],
     [24,26,56,99,99,99,99,99],
     [47,66,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99],
     [99,99,99,99,99,99,99,99]]
    ])

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
        qtT = np.stack([d.get_quantization_table(i) for i in range(2)])
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

        # jpeg-toolbox
        try:
            import jpeg_toolbox
        except Exception as e:
            logging.info(f"invalid installation of python-jpeg-toolbox: {e}")
            return
        img = jpeg_toolbox.load('examples/IMG_0791.jpeg')
        # process
        YT = img['coef_arrays'][0]\
            .reshape((1,int(img['image_height']/8),8,-1,8))
        YT = np.einsum('abcde->adbec', YT)
        CbCrT = np.stack([
            img['coef_arrays'][1].reshape((int(img['image_height']/8/2),8,-1,8)),
            img['coef_arrays'][2].reshape((int(img['image_height']/8/2),8,-1,8))
        ])
        CbCrT = np.einsum('abcde->adbec', CbCrT)
        qtT = img['quant_tables']
        #qtT = np.concatenate([img['quant_tables'], img['quant_tables'][1:]])

        # test quantization
        np.testing.assert_array_equal(qt, qtT)
        # test DCT coefficients
        np.testing.assert_array_equal(Y, YT)
        np.testing.assert_array_equal(CbCr, CbCrT)

    def test_dct(self):
        # write with different qt
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()
            im.write_dct("tmp/output.jpeg", Y, CbCr, qt)
        im = jpeglib.JPEG("tmp/output.jpeg")
        Y2,CbCr2,qt2 = im.read_dct()
        # test matrix
        np.testing.assert_array_equal(Y, Y2)
        np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(qt, qt2)
    
    def test_dct_qt(self):
        # write with different qt
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()
            im.write_dct("tmp/output.jpeg")
        im = jpeglib.JPEG("tmp/output.jpeg")
        Y2,CbCr2,qt2 = im.read_dct()
        # test matrix
        np.testing.assert_array_equal(Y, Y2)
        np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(qt, qt2)


    def test_dct_qt50(self):
        global qt50_standard

        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            _,_,_ = im.read_dct()
            im.write_dct("tmp/output.jpeg", qt = qt50_standard)
        
        im = jpeglib.JPEG("tmp/output.jpeg")
        _,_,qt50 = im.read_dct()

        # test matrix
        np.testing.assert_array_equal(qt50, qt50_standard)
    
    def test_dct_qt_edit(self):
        # write with different qt
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()
            qt[0,4,4] = 1 # change qt
            #qt[1,6,6] = 1 
            im.write_dct("tmp/output.jpeg", qt = qt)
        im = jpeglib.JPEG("tmp/output.jpeg")
        Y2,CbCr2,qt2 = im.read_dct()
        # test matrix
        np.testing.assert_array_equal(Y, Y2)
        np.testing.assert_array_equal(CbCr, CbCr2)
        np.testing.assert_array_equal(qt, qt2)

    def test_dct_quantized(self):
        # pass quantized qt
        im = jpeglib.JPEG("examples/IMG_0791.jpeg")
        Y_q,CbCr_q,qt_q = im.read_dct(quantized=True)
        im.write_dct("tmp/output.jpeg", quantized=True)
        # compare quantization
        Y,CbCr,qt = im.read_dct(quantized=False)
        # check quantized output
        np.testing.assert_array_equal(Y_q / qt_q[0], Y)
        np.testing.assert_array_equal(CbCr_q / qt_q[1], CbCr)
        np.testing.assert_array_equal(qt_q, qt)
        # compare quantization tables
        im1 = jpeglib.JPEG("examples/IMG_0791.jpeg")
        Y1,CbCr1,qt1 = im1.read_dct()
        im2 = jpeglib.JPEG("tmp/output.jpeg")
        Y2,CbCr2,qt2 = im2.read_dct()
        # test matrix
        np.testing.assert_array_equal(Y1, Y2)
        np.testing.assert_array_equal(CbCr1, CbCr2)
        np.testing.assert_array_equal(qt1, qt2)
    
    




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
    #     np.testing.assert_array_equal(qt, qtT.numpy()[:2])
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
    #         self._rainer_qt('Cr', qt[1])

__all__ = ["TestDCT"]