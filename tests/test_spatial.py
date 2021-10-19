
#import cv2
import logging
import numpy as np
import os
import pandas as pd
from PIL import Image
#from pylibjpeg import decode
import shutil
import subprocess
import sys
import unittest

sys.path.append('.')
import jpeglib


class TestSpatial(unittest.TestCase):
    def setUp(self):
        try: shutil.rmtree("tmp")
        except: pass
        finally: os.mkdir("tmp")
    def tearDown(self):
        shutil.rmtree("tmp")

    def _rainer_rgb(self, outchannel, rgb):

        # test DCT coefficients
        # get Rainer's (reference) result
        res = subprocess.call(
            "Rscript tests/test_rainer.R " + # script
            "rgb " + " " + # produce quantization table
            outchannel + " " +# Y, Cr or Cb
            "examples/IMG_0791.jpeg " + # input file
            "tmp/result.csv", # output file
            shell=True)
        if res != 0:
            raise Exception("Rainer's script failed!")
        df = pd.read_csv("tmp/result.csv", header=None)
        rgbRainer = df.to_numpy()
        # convert to 0-1
        rgb = rgb / 255

        # compare DCT coefficient matrices
        np.testing.assert_almost_equal(rgbRainer, rgb)

    def test_rainer_rgb_R(self):
        # read images
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            rgb = im.read_spatial(
                out_color_space="JCS_RGB",
                flags=['DO_FANCY_UPSAMPLING']
            )
            # call test
            self._rainer_rgb('R', rgb[:,:,0])
    
    def test_rainer_rgb_G(self):
        # read images
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            rgb = im.read_spatial(
                out_color_space="JCS_RGB",
                flags=['DO_FANCY_UPSAMPLING']
            )
            # call test
            self._rainer_rgb('G', rgb[:,:,1])

    def test_rainer_rgb_B(self):
        # read images
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            rgb = im.read_spatial(
                out_color_space="JCS_RGB",
                flags=['DO_FANCY_UPSAMPLING']
            )
            # call test
            self._rainer_rgb('B', rgb[:,:,2])


    def test_pil_read(self):
        # read rgb
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im1:
            x1 = im1.read_spatial(
                out_color_space='JCS_RGB',
                dct_method='JDCT_ISLOW',
                #dither_mode='JDITHER_NONE',
                flags=['DO_FANCY_UPSAMPLING','DO_BLOCK_SMOOTHING']
            )
            x1 = x1.squeeze()
        # read rgb via PIL
        im2 = Image.open("examples/IMG_0791.jpeg")
        x2 = np.array(im2)

        # test
        if (x1 != x2).any():
            logging.info("known PIL mismatch %.2f%%" % (((x2 - x1) != 0).mean()*100))
        else:
            np.testing.assert_almost_equal(x1, x2)

        # cleanup
        im2.close()
        
    # def test_cv2(self):
    #     # read rgb
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im1:
    #         x1 = im1.read_spatial(
    #             out_color_space='JCS_RGB',
    #             dct_method='JDCT_ISLOW'
    #         )
    #         x1 = x1.squeeze()
    #     # read rgb via cv2
    #     x2 = cv2.imread("examples/IMG_0791.jpeg", cv2.IMREAD_COLOR)
    #     x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB)
        
    #     # test
    #     if (x1 != x2).any():
    #         logging.info("known cv2 mismatch %.2f%%" % (((x2 - x1) != 0).mean()*100))
    #     else:
    #         np.testing.assert_almost_equal(x1, x2)
    #     # cleanup 
    #     im1.close()
    
    # def test_pylibjpeg(self):
    #     # read rgb
    #     with libjpeg.JPEG("examples/IMG_0791.jpeg") as im1:
    #         x1 = im1.read_spatial(flags=['DO_FANCY_UPSAMPLING'])
    #         x1 = x1.squeeze()
    #     # read rgb via pylibjpeg
    #     x2 = decode("examples/IMG_0791.jpeg")
        
    #     # test
    #     if (x1 != x2).any():
    #         logging.info("known pylibjpeg mismatch %.2f%%" % (((x2 - x1) != 0).mean()*100))
    #     else:
    #         np.testing.assert_almost_equal(x1, x2)

    def test_pil_write(self):
        q = 75 # quality

        # pass the image through jpeglib
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            data = im.read_spatial(flags=['DO_FANCY_UPSAMPLING'])
            im.write_spatial("tmp/test1.jpeg", data, quality=q, flags=['DO_FANCY_UPSAMPLING'])
    
        # pass the image through PIL
        im = Image.open("examples/IMG_0791.jpeg")
        im.save("tmp/test2.jpeg", quality=q, subsampling=-1)
        im.close()

        # load both with PIL
        im1 = Image.open("tmp/test1.jpeg")
        x1 = np.array(im1)
        im2 = Image.open("tmp/test2.jpeg")
        x2 = np.array(im2)
        
        # test
        if (x1 != x2).any():
            logging.info("known PIL recompression mismatch %.2f%%" % (((x2 - x1) != 0).mean()*100))
        else:
            np.testing.assert_almost_equal(x1, x2)

        # cleanup
        im1.close()
        im2.close()


    # def test_dct_pil(self):
    #     # setup
    #     try:
    #         shutil.rmtree("tmp")
    #     except:
    #         pass
    #     finally:
    #         os.mkdir("tmp")
    #     # pass the image through
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         Y,CbCr,qt = im.read_dct()
    #         im.write_dct("tmp/test.jpeg", Y, CbCr)
    #     # images
    #     im1 = Image.open("examples/IMG_0791.jpeg")
    #     im2 = Image.open("tmp/test.jpeg")
    #     # to numpy
    #     x1,x2 = np.array(im1),np.array(im2)
        
    #     # test
    #     np.testing.assert_almost_equal(x1, x2)

    #     # cleanup 
    #     shutil.rmtree("tmp")
    #     im1.close()
    #     im2.close()

    
    # #def test_pil_backwards(self):
    # #
    # #    # load image
    # #    im1 = jpeglib.JPEG("examples/IMG_0791.jpeg")
    # #    Y,CbCr,qt = im1.read_dct()
    # #    # reference
    # #    im2_rgb = Image.open("examples/IMG_0791.jpeg")
    # #
    # #    # convert reference to dct blocks
    # #    # TODO
    # #    im2_ycbcr = im2_rgb.convert('YCbCr')
    # #    data2 = np.array(im2_ycbcr)
    # #    blocks = data2.flatten()\
    # #        .reshape((3, int(data2.shape[0]/8), int(data2.shape[1]/8), 64))
    # #    im2_dct = np.apply_along_axis(dct, -1, blocks)
    # #    #print(im2_dct.shape)
    # #    #np.testing.assert_almost_equal(blocks)
    # #    #print(blocks.shape, Y.shape, CbCr.shape)
    # #    
    # #    # cleanup
    # #    im1.close()
    # #    im2_rgb.close()

    # def test_change_block1(self):
    #     # setup
    #     try:
    #         shutil.rmtree("tmp")
    #     except:
    #         pass
    #     finally:
    #         os.mkdir("tmp")
    #     # pass the image through
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         Y,CbCr,qt = im.read_dct()
    #         # change
    #         Y[0,0,0,:4]
    #         im.write_dct("tmp/test.jpeg", Y, CbCr)
    #     # images
    #     im1 = Image.open("examples/IMG_0791.jpeg")
    #     im2 = Image.open("tmp/test.jpeg")
    #     # to numpy
    #     x1,x2 = np.array(im1),np.array(im2)
        
    #     # test
    #     np.testing.assert_almost_equal(x1, x2)

    #     # cleanup 
    #     shutil.rmtree("tmp")
    #     im1.close()
    #     im2.close()

__all__ = ["TestSpatial"]