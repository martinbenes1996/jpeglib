Glossary
===================================

This part briefly explains the concepts behind the library.
The rest of documentation will refer to this page as a knowledge base.

.. contents:: Table of Contents
   :local:
   :depth: 1

JPEG compression
----------------

Description of JPEG compression


Discrete cosine transform
"""""""""""""""""""""""""

Description of discrete cosine transform

Forward DCT (also called DCT II)

.. math::
    Y_{uv}=\sqrt{\frac{2}{N}}\sqrt{\frac{2}{M}}\Lambda(u)\Lambda(v)\sum_{i=0}^{N-1}\sum_{j=0}^{M-1}\text{cos}\Big[\frac{\pi}{N}(i+.5)u\Big]\text{cos}\Big[\frac{\pi}{M}(j+.5)v\Big]X_{ij}

Inverse DCT (also called DCT III)

.. math::
    X_{ij}=\sqrt{\frac{2}{N}}\sqrt{\frac{2}{M}}\sum_{u=0}^{N-1}\sum_{v=0}^{M-1}\Lambda(u)\Lambda(v)\text{cos}\Big[\frac{\pi}{N}(i+.5)u\Big]\text{cos}\Big[\frac{\pi}{M}(j+.5)v\Big]Y_{uv}


References for JPEG compression
"""""""""""""""""""""""""""""""

* `Computerphile: JPEG DCT, Discrete Cosine Transform <https://www.youtube.com/watch?v=Q2aEzeMDHMA&ab_channel=Computerphile>`_
* `The Discrete Cosine Transform (DCT) <https://users.cs.cf.ac.uk/Dave.Marshall/Multimedia/node231.html>`_

libjpeg C library
-----------------

Description of libjpeg

Mention different version

JPEG sampling factor
""""""""""""""""""""

As human eyes are better in seeing light intensity (*luminance*) than colors (*chrominance*),
chrominance is another component that can be reduced with so called chroma subsampling.
It is controlled using sampling factors, a pair of integer numbers per each channel,
3 pairs for YCbCr, `[[h1,w1],[h2,w2],[h3,w3]]`.

Let's have an example: an RGB image with 300px x 300px and sampling factors [[2,3],[1,2],[2,1]].
What we want is the resolution of channels YCbCr stored in JPEG. We start with computing
hmax and wmax (2 and 3) and normalizing factors with the respective maximum; now we have
[[1,1],[1/2,2/3],[1,1/3]], which stands for how large are sides compared to a rectangle 2x3 pixels.
Thus multiplied by the image size, Y has 300 x 300, Cb 150 x 200 and Cr 300 x 100.

You can specify the chroma subsampling before writing a JPEG image as follows:

>>> # im is a jpeglib.JPEG object
>>>
>>> # Variant 1: Specify in J:a:b notation
>>> im.samp_factor = '4:4:4'
>>>
>>> # Variant 2: Specify per-channel sampling factors as a list [[Y_h, Y_v], [Cb_h, Cb_v],[Cr_h, Cr_v]]
>>> im.samp_factor = [[1, 1], [1, 1], [1, 1]]
>>>
>>> # now you can write im to a file

Flags
"""""

Compression and decompression parameters have a crucial impact on the output JPEG.
Boolean parameters are in jpeglib simply called flags, and contain following.

**Compression**

- DO_FANCY_SAMPLING, DO_FANCY_DOWNSAMPLING
- OPTIMIZE_CODING
- PROGRESSIVE_MODE
- ARITH_CODE
- TRELLIS_QUANT (>= mozjpeg300)
- TRELLIS_QUANT_DC (>= mozjpeg300)
- FORCE_BASELINE
- WRITE_JFIF_HEADER
- WRITE_ADOBE_MARKER

**Decompression**

- DO_FANCY_UPSAMPLING
- QUANTIZE_COLORS
- DO_BLOCK_SMOOTHING
- TWO_PASS_QUANTIZE
- ENABLE_1PASS_QUANT
- ENABLE_2PASS_QUANT
- ENABLE_EXTERNAL_QUANT
- CCIR601_SAMPLING

The flags can be specified as a list of string, either enabling or disabling the option.
Following code decompresses the input using simple upsampling, and then compresses it again
into progressive JPEG, with explicitly disabling Huffman code optimization.

>>> jpeglib.version.set('6b')
>>> im = jpeglib.read_spatial("input.jpeg", flags=["-DO_FANCY_UPSAMPLING"])
>>> im.write_spatial("output.jpeg", flags=["+PROGRESSIVE_MODE", "-OPTIMIZE_CODING"])

The values of not-specified flags are kept to be defaultly set by the selected libjpeg version,
or copied from the source image.


References
""""""""""

* `Using the IJG JPEG library <https://freedesktop.org/wiki/Software/libjpeg/>`_
* `Interface Definitions for libjpeg <https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpegman.html>`_
* `Data Definitions for libjpeg <https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpeg-ddefs.html>`_
* `Chroma subsampling and JPEG sampling factors <https://zpl.fi/chroma-subsampling-and-jpeg-sampling-factors/>`_


Progressive JPEG
----------------

Progressive JPEG arranges the data in the file by placing the low-level image first, and details later.
On slow internet connection, progressive JPEG loads by gradually focusing, while sequential JPEG shows in full quality line-by-line.

Progressive JPEG consists of scans, which carry parts of the DCT coefficients. DCT coefficients can be split by subband (frequency) and by precision (bits).
After full loading of all the scans, progressive image should be, in theory, identical to its sequential counterpart.
However, MozJPEG uses Trellis optimization which optimizes the file size and allows introduction of a imperceptible distortion.

References
""""""""""

* `Hofer, Böhme: Progressive JPEGs in the Wild: Implications for Information Hiding and Forensics <https://informationsecurity.uibk.ac.at/pdfs/HB2023_IHMMSEC.pdf>`_



Glossary terms
--------------

.. glossary::

    DCT
        Discrete cosine transform

    libjpeg
        C library developed by IJC

    JPEG
        Joint Photographic Experts Group, image compression standard.

    JPG
        Synonym to :term:`JPEG`.

    spatial domain
        Description of spatial domain


