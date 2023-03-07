Usage
=====

.. contents:: Table of Contents
   :local:
   :depth: 1

Installation and setup
----------------------

To use ``jpeglib``, first install it using pip:

.. code-block:: console

   $ pip3 install jpeglib

Import the package with

>>> import jpeglib

.. warning::

   Compression comes with loss and thus different jpeg implementations might
   produce slightly different results. The library *jpeglib* brings
   solution as being able to select the library in runtime.

   To better address this, jpeglib comes with `unit tests <https://github.com/martinbenes1996/jpeglib/actions/workflows/unittests_on_commit.yml>`_,
   where you can check, how different the outputs are compared to other popular
   Python packages.

You can specify a particular :term:`libjpeg` version to use with
:py:func:`set_libjpeg_version()`.

>>> jpeglib.version.set('6b')

Currently supported are
* libjpeg versions ``"6b"``, ``"7"``, ``"8"``, ``"8a"``, ``"8b"``, ``"8c"``, ``"8d"``, ``"9"``, ``"9a"``, ``"9b"``, ``"9c"``, ``"9d"`` , ``"9e"``,
* libjpeg-turbo versions ``"turbo120"``, ``"turbo130"``, ``"turbo140"``, ``"turbo150"``, ``"turbo200"``, ``"turbo210"``, and
* mozjpeg versions ``"mozjpeg101"``, ``"mozjpeg201"``, ``"mozjpeg300"``, and ``"mozjpeg403"``.

.. note::

   Feel free to switch to supported libjpeg-turbo versions, but do not rely that all SIMD extensions,
   supported by libjpeg-turbo, are used.
   An effort has been made to integrate libjpeg-turbo, but SIMD configuration is rather complex,
   so I cannot guarrantee full support at this moment.

   To see that SIMD is used at all, I measured the time of execution and compared it to libjpeg.
   Via jpeglib on M1 Chip (Neon), libjpeg-turbo is significantly faster both in compression and decompression.

Spatial domain
--------------

In :term:`JPEG`, the pixel data (such as RGB) in so called spatial domain are compressed and to get them,
the image has to be decompressed. Similarly to save pixels as JPEG, they are compressed.

Reading the spatial domain
^^^^^^^^^^^^^^^^^^^^^^^^^^

Decompress input file ``input.jpeg`` into spatial representation in numpy array with

>>> im = jpeglib.read_spatial("input.jpeg")
>>> im.spatial


The output channels depend on the source file. You can explicitly request returning RGB

>>> im = jpeglib.read_spatial("input.jpeg", out_color_space=jpeglib.Colorspace.JCS_RGB)
>>> rgb = im.spatial

For more parameters check out the documentation of the function `jpeglib.JPEG.read_spatial <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.functional.read_spatial>`_
and the class of the returned object `jpeglib.spatial_jpeg.SpatialJPEG <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.spatial_jpeg.SpatialJPEG>`_

Writing the spatial domain
^^^^^^^^^^^^^^^^^^^^^^^^^^

Compression of a spatial domain to an output file ``output.jpeg`` is done with

>>> im.write_spatial("output.jpeg")

Compression parameters connected with the JPEG, such as dimensions, colorspace or markers
are attributes of the object and can be overwritten. Others are parameters of the function.

>>> im.samp_factor = ((2,1),(1,1),(1,1))
>>> im.write_spatial("output.jpeg", dct_method=jpeglib.DCTMethod.JDCT_IFAST)

The color space is chosen based on reading. All the parameter options are listen in the
`jpeglib.spatial_jpeg.SpatialJPEG.write_spatial <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.spatial_jpeg.SpatialJPEG.write_spatial>`_
documentation.

DCT coefficients
----------------

:term:`DCT` (*Discrete cosine transform*) is one of the steps during JPEG compression and decompression.
Read more about it in `JPEG compression glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#jpeg-compression>`_.

Unlike spatial domain writing, reading and writing of DCT coefficients is lossless.

Reading the DCT coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Acquire the quantized DCT coefficients of an input file ``input.jpeg`` with

>>> im = jpeglib.read_dct("input.jpeg")
>>> im.Y; im.Cb; im.Cr; im.qt

The four members are tensors of luminance (Y) and chrominance (Cb, Cr) DCT coefficients and
quantization tables (qt). Read more information in the `jpeglib.functional.read_dct <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.functional.read_dct>`_
documentation.

To get dequantized DCT coefficients, multiply the tensors by quantization table.

>>> Y_deq = im.Y * im.qt[0]
>>> Cb_deq = im.Cb * im.qt[1]
>>> Cr_deq = im.Cr * im.qt[2]

Writing the DCT coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write the quantized coefficients to an output file ``output.jpeg`` with

>>> im.write_dct("output.jpeg")

The function reference can be found in the `jpeglib.dct_jpeg.DCTJPEG.write_dct <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.dct_jpeg.DCTJPEG.write_dct>`_
documentation.

jpegio format
^^^^^^^^^^^^^

Existing package jpegio already offers interface to work with DCT coefficients and quantization tables.
To make an easy transition to jpeglib, we offer an simple abstraction of the jpegio interface.

>>> im = jpeglib.read_dct("input.jpeg")
>>> im = jpeglib.to_jpegio(im)
>>> im.coef_arrays[0][:8,-8:]   # -> im.Y[0,-1]
>>> im.coef_arrays[1][-8:,8:16] # -> im.Cr[-1,1]
>>> im.quant_tables[0]          # -> im.qt[0]
