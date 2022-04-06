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

References for JPEG compression
"""""""""""""""""""""""""""""""

* `Computerphile: JPEG DCT, Discrete Cosine Transform <https://www.youtube.com/watch?v=Q2aEzeMDHMA&ab_channel=Computerphile>`_

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

Discrete cosine transform
"""""""""""""""""""""""""

Description of discrete cosine transform

References for libjpeg
""""""""""""""""""""""

* `Using the IJG JPEG library <https://freedesktop.org/wiki/Software/libjpeg/>`_
* `Interface Definitions for libjpeg <https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpegman.html>`_
* `Data Definitions for libjpeg <https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpeg-ddefs.html>`_
* `Chroma subsampling and JPEG sampling factors <https://zpl.fi/chroma-subsampling-and-jpeg-sampling-factors/>`_

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
    
    
