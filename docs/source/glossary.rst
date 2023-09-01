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

An RGB image with 300px x 300px compressed with sampling factors [[2,2],[2,1],[1,2]]
will have components Y of size 200 x 200, Cb of size 200 x 100 and Cr of size 100 x 200.
You can specify this before writing a JPEG image

>>> # im is a jpeglib.JPEG object
>>>
>>> # Sampling factors as a list [[Y_v, Y_h], [Cb_v, Cb_h],[Cr_v, Cr_h]]
>>> im.samp_factor = [[2, 2], [2, 1], [1, 2]]
>>>
>>> # now you can write im to a file

Researchers often use J:a:b notation, which is shorter, but assumes same subsampling for both chrominances.
No chroma subsampling, [[1,1],[1,1],[1,1]], is denoted as 4:4:4.

>>> # Variant 1: Per-channel sampling factors
>>> im.samp_factor = [[1, 1], [1, 1], [1, 1]]
>>>
>>> # Variant 2: Specify in J:a:b notation
>>> im.samp_factor = '4:4:4'

Following table contains pairs of sampling factor in per-channel and J:a:b notation.

.. list-table:: Notations for chroma sampling.
   :widths: 25 50 25
   :header-rows: 1

   * - J:a:b notation
     - Per-channel notation
     - Applied ratio
   * - 4:4:4
     - [[1, 1], [1, 1], [1, 1]]
     - :math:`1/1`, :math:`1/1`

.. note::

    For consistence with the rest of interface, jpeglib uses vertical-horizontal order.
    In cjpeg, ImageMagick, and `this tutorial <https://zpl.fi/chroma-subsampling-and-jpeg-sampling-factors/>`_,
    the chroma sampling factors are defined in horizontal-vertical order.

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

**References**

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

**References**

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


