Usage
=====

.. contents:: Table of Contents

Installation and setup
----------------------

To use ``jpeglib``, first install it using pip:

.. code-block:: console

   $ pip3 install jpeglib

Import the package with

>>> import jpeglib

You can specify a particular libjpeg to use with
:py:func:`set_libjpeg_version()`.

>>> jpeglib.version.set('6b')

Currently supported versions are ``"6b"`` and ``"8d"``. 

Pixel data
----------

Reading
^^^^^^^

Decompress input file ``input.jpeg`` into spatial representation in numpy array with

```python
im = jpeglib.JPEG("input.jpeg")
spatial = im.read_spatial()
```

The output channels depend on the source file. You can explicitly request returning RGB

```python
im = jpeglib.JPEG("input.jpeg")
rgb = im.read_spatial(output_color_space='JCS_RGB')
```

For more parameters check out the :ref:`documentation <jpeglib.JPEG.read_spatial>`.

Writing
^^^^^^^

Compression of a numpy array to an output file ``output.jpeg`` is done with

```python
spatial = im.read_spatial("output.jpeg", spatial)
```

The color space is chosen based on reading. All the parameter options are listen in the
:ref:`documentation <jpeglib.JPEG.write_spatial>`.

DCT coefficients
----------------

Get DCT coefficients and quantization matrix

TODO