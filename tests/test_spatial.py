
import logging
import numpy as np
import os
from parameterized import parameterized
from PIL import Image
import tempfile
import unittest

import jpeglib
from _defs import ALL_VERSIONS, version_cluster

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
     [99, 99, 99, 99, 99, 99, 99, 99]],
])


class TestSpatial(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.original_version = jpeglib.version.get()
        self.tmp = tempfile.NamedTemporaryFile(suffix='.jpeg', delete=False)
        self.tmp.close()

    def tearDown(self):
        os.remove(self.tmp.name)
        del self.tmp
        jpeglib.version.set(self.original_version)


    def assert_compressed(self, x1, x2):
        mse = np.mean((
            x1/255. -
            x2/255.
        )**2)
        self.assertLessEqual(mse, 1.)

    def test_spatial(self):
        """Test of compression and decompression."""
        self.logger.info("test_spatial")
        # get decompressed spatial
        im = jpeglib.read_spatial("examples/IMG_0311.jpeg")
        # write and read again
        im.write_spatial(self.tmp.name)
        im2 = jpeglib.read_spatial(self.tmp.name)
        # test compressed version
        self.assert_compressed(im.spatial, im2.spatial)

    def test_synthetic_spatial(self):
        """Test of synthetic image being generated and stored.

        Comparison is being done based on MSE lower than 1.
        (weak, but better than nothing)
        """
        self.logger.info("test_synthetic_spatial")
        # create synthetic JPEG
        np.random.seed(12345)
        spatial = (np.random.rand(512,512,3)*255).astype(np.int16)
        jpeglib.from_spatial(
            spatial=spatial,
            in_color_space='JCS_RGB'
        ).write_spatial(self.tmp.name)
        # load and compare
        im = jpeglib.read_spatial(self.tmp.name)
        # test compressed version
        self.assert_compressed(spatial, im.spatial)


    @parameterized.expand(ALL_VERSIONS)
    # @parameterized.expand(LIBJPEG_VERSIONS)
    def test_default_quality(self, version):
        """Test default quality factor for a given version."""
        self.logger.info(f"test_default_quality_{version}")
        with jpeglib.version(version):
            jpeg = jpeglib.read_spatial("examples/IMG_0311.jpeg")
            # with explicit qf
            jpeg.write_spatial(self.tmp.name, qt=75, base_quant_tbl_idx=0)
            _, _, qt1 = jpeglib.read_dct(self.tmp.name).load()
            # with default qf
            jpeg.write_spatial(self.tmp.name)
            _, _, qt2 = jpeglib.read_dct(self.tmp.name).load()
            # test equal qts
            np.testing.assert_array_equal(qt1, qt2)


    # def test_spatial_quality(self):
    #     global qt50_standard
    #     # print("test_spatial_quality")
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         x = im.read_spatial()
    #         im.write_spatial(x, self.tmp.name, qt=50)
    #     im = jpeglib.JPEG(self.tmp.name)
    #     _, _, qt50 = im.read_dct()

    #     # test matrix
    #     np.testing.assert_array_equal(qt50, qt50_standard)

    @parameterized.expand([
        ['6b', '7', True],
        ['9', '9a', False],
    ])
    def test_mismatch_baseline(self, v1, v2, equal_Y):
        """Decompress with given two versions and observe differing output."""
        self.logger.info(f"test_mismatch_baseline_{v1}_{v2}")
        # decompress image with given versions
        with jpeglib.version(v1):
            x1 = jpeglib.read_spatial("examples/IMG_0311.jpeg").spatial
        with jpeglib.version(v2):
            x2 = jpeglib.read_spatial("examples/IMG_0311.jpeg").spatial
        # compare not equal
        self.assertFalse((x1 == x2).all())

    @parameterized.expand([
        ['6b','turbo120','turbo130','turbo140','turbo150','turbo200','turbo210'],
        ['7','8','8a','8b','8c','8d','9'],
        ['9a','9b','9c','9d','9e'],
        ['6b','mozjpeg101','mozjpeg201','mozjpeg300','mozjpeg403'],
    ], name_func=version_cluster)
    def test_equal_baseline(self, *versions):
        """Decompress with given versions and observe the same output."""
        self.logger.info(f"test_mismatch_baseline_{'_'.join(versions)}")
        # decompress image with reference
        with jpeglib.version(versions[0]):
            x_ref = jpeglib.read_spatial("examples/IMG_0311.jpeg").spatial
        # decompress with each version and compare to the reference
        for v in versions:
            with jpeglib.version(v):
                x = jpeglib.read_spatial("examples/IMG_0311.jpeg").spatial
            np.testing.assert_array_equal(x_ref, x)

    # def test_spatial_qt(self):
    #     global qt50_standard
    #     # print("test_spatial_qt")
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         x = im.read_spatial()
    #         im.write_spatial(x, self.tmp.name, qt=qt50_standard)

    #     im = jpeglib.JPEG(self.tmp.name)
    #     _, _, qt50 = im.read_dct()

    #     # test matrix
    #     np.testing.assert_array_equal(qt50, qt50_standard)

    # def test_pil_read(self):
    #     jpeglib.version.set('8d')
    #     # read rgb
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im1:
    #         x1 = im1.read_spatial(
    #             out_color_space='JCS_RGB',
    #             dct_method='JDCT_ISLOW',
    #             # dither_mode='JDITHER_NONE',
    #             flags=['DO_FANCY_UPSAMPLING', 'DO_BLOCK_SMOOTHING']
    #         )
    #         x1 = x1.squeeze()
    #     # read rgb via PIL
    #     im2 = Image.open("examples/IMG_0791.jpeg")
    #     x2 = np.array(im2)

    #     # test
    #     if (x1 != x2).any():
    #         logging.info(
    #             "known PIL mismatch %.2f%% (%d)" % (
    #                 ((x2 - x1) != 0).mean()*100, ((x2 - x1) != 0).sum()
    #             )
    #         )
    #     else:
    #         np.testing.assert_almost_equal(x1, x2)

    #     # cleanup
    #     im2.close()

    # def test_pil_write(self):
    #     # print("test_pil_write")
    #     q = 75  # quality
    #     # pass the image through jpeglib
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         # print("read spatial")
    #         data = im.read_spatial(flags=['DO_FANCY_UPSAMPLING'])
    #         # print("write spatial")
    #         im.write_spatial(data, self.tmp.name, qt=q,
    #                          flags=['DO_FANCY_UPSAMPLING'])
    #     # pass the image through PIL
    #     im = Image.open("examples/IMG_0791.jpeg")
    #     im.save(self.tmp2.name, qt=q, subsampling=-1)
    #     im.close()
    #     # load both with PIL
    #     im1 = Image.open(self.tmp.name)
    #     x1 = np.array(im1)
    #     im2 = Image.open(self.tmp2.name)
    #     x2 = np.array(im2)

    #     # test
    #     if (x1 != x2).any():
    #         logging.info(
    #             "known PIL recompression mismatch %.2f%%" % (
    #                 ((x2 - x1) != 0).mean()*100
    #             )
    #         )
    #     else:
    #         np.testing.assert_almost_equal(x1, x2)

    #     # cleanup
    #     im1.close()
    #     im2.close()

    # def test_only_write(self):
    #     # generate random image
    #     x = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    #     # write into file
    #     with jpeglib.JPEG(self.tmp.name) as im:
    #         im.write_spatial(x)

    # # def _rainer_rgb(self, outchannel, rgb):

    # #     # test DCT coefficients
    # #     # get Rainer's (reference) result
    # #     res = subprocess.call(
    # #         "Rscript tests/test_rainer.R " + # script
    # #         "rgb " + " " + # produce quantization table
    # #         outchannel + " " +# Y, Cr or Cb
    # #         "examples/IMG_0791.jpeg " + # input file
    # #         "tmp/result.csv", # output file
    # #         shell=True)
    # #     if res != 0:
    # #         raise Exception("Rainer's script failed!")
    # #     df = pd.read_csv("tmp/result.csv", header=None)
    # #     rgbRainer = df.to_numpy()
    # #     # convert to 0-1
    # #     rgb = rgb / 255

    # #     # compare DCT coefficient matrices
    # #     np.testing.assert_almost_equal(rgbRainer, rgb)

    # # def test_rainer_rgb_R(self):
    # #     # read images
    # #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    # #         rgb = im.read_spatial(
    # #             out_color_space="JCS_RGB",
    # #             flags=['DO_FANCY_UPSAMPLING']
    # #         )
    # #         # call test
    # #         self._rainer_rgb('R', rgb[:,:,0])

    # # def test_rainer_rgb_G(self):
    # #     # read images
    # #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    # #         rgb = im.read_spatial(
    # #             out_color_space="JCS_RGB",
    # #             flags=['DO_FANCY_UPSAMPLING']
    # #         )
    # #         # call test
    # #         self._rainer_rgb('G', rgb[:,:,1])

    # # def test_rainer_rgb_B(self):
    # #     # read images
    # #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    # #         rgb = im.read_spatial(
    # #             out_color_space="JCS_RGB",
    # #             flags=['DO_FANCY_UPSAMPLING']
    # #         )
    # #         # call test
    # #         self._rainer_rgb('B', rgb[:,:,2])

    # # def test_cv2(self):
    # #     jpeglib.version.set('8d')
    # #     # read rgb
    # #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im1:
    # #         x1 = im1.read_spatial(
    # #             out_color_space='JCS_RGB',
    # #             dct_method='JDCT_ISLOW'
    # #         )
    # #         x1 = x1.squeeze()
    # #     # read rgb via cv2
    # #     x2 = cv2.imread("examples/IMG_0791.jpeg", cv2.IMREAD_COLOR)
    # #     x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB)

    # #     # test
    # #     if (x1 != x2).any():
    # #         logging.info("known cv2 mismatch %.2f%%" % (((x2 - x1) != 0).mean()*100))  # noqa: E501
    # #     else:
    # #         np.testing.assert_almost_equal(x1, x2)
    # #     # cleanup
    # #     im1.close()

    # # def test_pylibjpeg(self):
    # #     # read rgb
    # #     with libjpeg.JPEG("examples/IMG_0791.jpeg") as im1:
    # #         x1 = im1.read_spatial(flags=['DO_FANCY_UPSAMPLING'])
    # #         x1 = x1.squeeze()
    # #     # read rgb via pylibjpeg
    # #     x2 = decode("examples/IMG_0791.jpeg")

    # #     # test
    # #     if (x1 != x2).any():
    # #         logging.info("known pylibjpeg mismatch %.2f%%" % (((x2 - x1) != 0).mean()*100))  # noqa: E501
    # #     else:
    # #         np.testing.assert_almost_equal(x1, x2)

    # # def test_dct_pil(self):
    # #     # setup
    # #     try:
    # #         shutil.rmtree("tmp")
    # #     except:
    # #         pass
    # #     finally:
    # #         os.mkdir("tmp")
    # #     # pass the image through
    # #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    # #         Y,CbCr,qt = im.read_dct()
    # #         im.write_dct("tmp/test.jpeg", Y, CbCr)
    # #     # images
    # #     im1 = Image.open("examples/IMG_0791.jpeg")
    # #     im2 = Image.open("tmp/test.jpeg")
    # #     # to numpy
    # #     x1,x2 = np.array(im1),np.array(im2)

    # #     # test
    # #     np.testing.assert_almost_equal(x1, x2)

    # #     # cleanup
    # #     shutil.rmtree("tmp")
    # #     im1.close()
    # #     im2.close()

    # # #def test_pil_backwards(self):
    # # #
    # # #    # load image
    # # #    im1 = jpeglib.JPEG("examples/IMG_0791.jpeg")
    # # #    Y,CbCr,qt = im1.read_dct()
    # # #    # reference
    # # #    im2_rgb = Image.open("examples/IMG_0791.jpeg")
    # # #
    # # #    # convert reference to dct blocks
    # # #    # TODO
    # # #    im2_ycbcr = im2_rgb.convert('YCbCr')
    # # #    data2 = np.array(im2_ycbcr)
    # # #    blocks = data2.flatten()\
    # # #        .reshape((3, int(data2.shape[0]/8), int(data2.shape[1]/8), 64))
    # # #    im2_dct = np.apply_along_axis(dct, -1, blocks)
    # # #    #print(im2_dct.shape)
    # # #    #np.testing.assert_almost_equal(blocks)
    # # #    #print(blocks.shape, Y.shape, CbCr.shape)
    # # #
    # # #    # cleanup
    # # #    im1.close()
    # # #    im2_rgb.close()

    # def test_change_block1(self):
    #     # pass the image through
    #     with jpeglib.JPEG("examples/IMG_0791.jpeg") as im:
    #         Y, CbCr, qt = im.read_dct()
    #         # change
    #         Y[0, 0, 0, :4] = 0
    #         im.write_dct(Y, CbCr, self.tmp.name)
    #     # images
    #     im1 = Image.open("examples/IMG_0791.jpeg")
    #     im2 = Image.open(self.tmp.name)
    #     # to numpy
    #     x1, x2 = np.array(im1), np.array(im2)

    #     D = np.abs(x1 - x2)
    #     np.testing.assert_raises(
    #         AssertionError,
    #         np.testing.assert_array_equal,
    #         D[:8, :8],
    #         np.zeros((8, 8, 3))
    #     )
    #     D[:8, :8] = 0
    #     np.testing.assert_array_almost_equal(D, np.zeros(D.shape))
    #     # D[D != 0] = 255
    #     # print(D[:8,:8,:])

    #     # import matplotlib.pyplot as plt
    #     #
    #     # plt.imshow(D)
    #     # plt.show()

    # #     # test
    # #     np.testing.assert_almost_equal(x1, x2)


__all__ = ["TestSpatial"]
