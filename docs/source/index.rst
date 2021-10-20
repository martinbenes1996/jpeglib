jpeglib
===================================

**jpeglib** is a Python package, envelope for the popular C library *libjpeg*
for handling JPEG files.

Currently almost all of the popular Python image libraries use *libjpeg* under the hood,
however do not expose the whole spectrum of parameters that libjpeg offers.
At the same time it is usually also impossible to load a low-level JPEG components - DCT coefficients
and quantization tables. All of this is possible with **jpeglib**.

Getting the DCT coefficients with jpeglib is as simple as

>>> import jpeglib
>>> with jpeglib.JPEG("input.jpeg") as im:
>>>     Y,CbCr,qt = im.read_dct()

.. note::

   This project is under active development.


Contents
--------

.. toctree::
   :maxdepth: 2
   
   usage
   reference
   glossary