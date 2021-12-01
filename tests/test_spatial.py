
import logging
import numpy as np
import os
from PIL import Image
import shutil
import sys
import unittest

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

class TestSpatial(unittest.TestCase):        
    def setUp(self):
        try: shutil.rmtree("tmp")
        except: pass
        finally: os.mkdir("tmp")
    # def tearDown(self):
    #     shutil.rmtree("tmp")
        
    def _test_synthetic(self, color_space=None):
        global qt50_standard
        # generate uniform image
        x_in = np.ones((256, 256, 3), np.uint8)*255
        # compress
        im = jpeglib.JPEG()
        output_file = f"tmp/output_{color_space}.jpeg"
        im.write_spatial(output_file, x_in, in_color_space=color_space)
        # decompress
        im = jpeglib.JPEG(output_file)
        x_out = im.read_spatial(out_color_space=color_space)

        # test matrix
        np.testing.assert_array_equal(x_in, x_out)

    def test_synthetic_default(self):
        self._test_synthetic()
    def test_synthetic_rgb(self):
        self._test_synthetic(color_space='JCS_RGB')
    def test_synthetic_ycbcr(self):
        self._test_synthetic(color_space='JCS_YCbCr')

    def test_spatial_quality(self):
        global qt50_standard
        #print("test_spatial_quality")
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            _ = im.read_spatial()
            im.write_spatial("tmp/output.jpeg", qt=50)
        im = jpeglib.JPEG("tmp/output.jpeg")
        _,_,qt50 = im.read_dct()

        # test matrix
        np.testing.assert_array_equal(qt50, qt50_standard)

    def test_spatial_qt(self):
        global qt50_standard
        #print("test_spatial_qt")
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            _ = im.read_spatial()
            im.write_spatial("tmp/output.jpeg", qt=qt50_standard)
        
        im = jpeglib.JPEG("tmp/output.jpeg")
        _,_,qt50 = im.read_dct()

        # test matrix
        np.testing.assert_array_equal(qt50, qt50_standard)

    def test_pil_read(self):
        jpeglib.version.set('8d')
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
            logging.info("known PIL mismatch %.2f%% (%d)" % (((x2 - x1) != 0).mean()*100, ((x2 - x1) != 0).sum()))
        else:
            np.testing.assert_almost_equal(x1, x2)

        # cleanup
        im2.close()

    def test_pil_write(self):
        #print("test_pil_write")
        q = 75 # quality
        # pass the image through jpeglib
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            #print("read spatial")
            data = im.read_spatial(flags=['DO_FANCY_UPSAMPLING'])
            #print("write spatial")
            im.write_spatial("tmp/test1.jpeg", data, qt=q, flags=['DO_FANCY_UPSAMPLING'])
        # pass the image through PIL
        im = Image.open("examples/IMG_0791.jpeg")
        im.save("tmp/test2.jpeg", qt=q, subsampling=-1)
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








    # def _rainer_rgb(self, outchannel, rgb):

    #     # test DCT coefficients
    #     # get Rainer's (reference) result
    #     res = subprocess.call(
    #         "Rscript tests/test_rainer.R " + # script
    #         "rgb " + " " + # produce quantization table
    #         outchannel + " " +# Y, Cr or Cb
    #         "examples/IMG_0791.jpeg " + # input file
    #         "tmp/result.csv", # output file
    #         shell=True)
    #     if res != 0:
    #         raise Exception("Rainer's script failed!")
    #     df = pd.read_csv("tmp/result.csv", header=None)
    #     rgbRainer = df.to_numpy()
    #     # convert to 0-1
    #     rgb = rgb / 255

    #     # compare DCT coefficient matrices
    #     np.testing.assert_almost_equal(rgbRainer, rgb)

    # def test_rainer_rgb_R(self):
    #     # read images
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         rgb = im.read_spatial(
    #             out_color_space="JCS_RGB",
    #             flags=['DO_FANCY_UPSAMPLING']
    #         )
    #         # call test
    #         self._rainer_rgb('R', rgb[:,:,0])
    
    # def test_rainer_rgb_G(self):
    #     # read images
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         rgb = im.read_spatial(
    #             out_color_space="JCS_RGB",
    #             flags=['DO_FANCY_UPSAMPLING']
    #         )
    #         # call test
    #         self._rainer_rgb('G', rgb[:,:,1])

    # def test_rainer_rgb_B(self):
    #     # read images
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         rgb = im.read_spatial(
    #             out_color_space="JCS_RGB",
    #             flags=['DO_FANCY_UPSAMPLING']
    #         )
    #         # call test
    #         self._rainer_rgb('B', rgb[:,:,2])



        
    # def test_cv2(self):
    #     jpeglib.version.set('8d')
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

    def test_change_block1(self):
        # pass the image through
        with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
            Y,CbCr,qt = im.read_dct()
            # change
            Y[0,0,0,:4] = 0
            im.write_dct("tmp/test.jpeg", Y, CbCr)
        # images
        im1 = Image.open("examples/IMG_0791.jpeg")
        im2 = Image.open("tmp/test.jpeg")
        # to numpy
        x1,x2 = np.array(im1),np.array(im2)
        
        D = np.abs(x1 - x2)
        np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_equal,
            D[:8,:8,:],
            np.zeros((8,8,3))
        )
        D[:8,:8,:] = 0
        np.testing.assert_array_almost_equal(D, np.zeros(D.shape))
        #D[D != 0] = 255
        #print(D[:8,:8,:])

        #import matplotlib.pyplot as plt
        #
        #plt.imshow(D)
        #plt.show()
        
    #     # test
    #     np.testing.assert_almost_equal(x1, x2)


__all__ = ["TestSpatial"]