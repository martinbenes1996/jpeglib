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

You can specify a particular :term:`libjpeg` version to use with
:py:func:`set_libjpeg_version()`.

>>> jpeglib.version.set('6b')

Currently supported versions are ``"6b"`` and ``"8d"``. 

Pixel data
----------

In :term:`JPEG`, the pixel data (such as RGB) are compressed and to get them,
the image has to be decompressed. Similarly to save pixels as JPEG,
they are compressed.

.. warning::
   
   Compression comes with loss and thus different implementations might
   produce slightly different results.
   
   To better address this, jpeglib comes with `unit tests <https://github.com/martinbenes1996/jpeglib/actions/workflows/unittests_on_commit.yml>`_,
   where you can check, how different the outputs are compared to other popular
   Python packages.

Reading the pixel data
^^^^^^^^^^^^^^^^^^^^^^

Decompress input file ``input.jpeg`` into spatial representation in numpy array with

>>> im = jpeglib.JPEG("input.jpeg")
>>> spatial = im.read_spatial()


The output channels depend on the source file. You can explicitly request returning RGB

>>> im = jpeglib.JPEG("input.jpeg")
>>> rgb = im.read_spatial(output_color_space='JCS_RGB')


For more parameters check out the `jpeglib.JPEG.read_spatial <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.JPEG.read_spatial>`_
documentation.

Writing the pixel data
^^^^^^^^^^^^^^^^^^^^^^

Compression of a numpy array to an output file ``output.jpeg`` is done with

>>> im.write_spatial("output.jpeg", spatial)

The color space is chosen based on reading. All the parameter options are listen in the
`jpeglib.JPEG.write_spatial <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.JPEG.write_spatial>`_
documentation.

DCT coefficients
----------------

:term:`DCT` (*Discrete cosine transform*) is one of the steps during JPEG compression and decompression.
Read more about it in `JPEG compression glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#jpeg-compression>`_.

Unlike spatial domain writing, reading and writing of quantized DCT coefficients is lossless.

.. note::
   
   Package jpeglib reads and writes quantized DCT coefficients (C type ``short``).


Reading the DCT coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Acquire the quantized DCT coefficients of an input file ``input.jpeg`` with

>>> Y,CbCr,qt = im.read_dct()

What you recieve is tensors of luminance and chrominance DCT coefficients and
quantization tables, read more specific information in the `jpeglib.JPEG.read_dct <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.JPEG.read_dct>`_
documentation.

Writing the DCT coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write the quantized coefficients to an output file ``output.jpeg`` with

>>> im.write_dct("output.jpeg", Y, CbCr)

The function reference can be found in the `jpeglib.JPEG.write_dct <https://jpeglib.readthedocs.io/en/latest/reference.html#jpeglib.JPEG.write_dct>`_ 
documentation.

