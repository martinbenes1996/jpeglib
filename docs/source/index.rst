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
>>> im = jpeglib.read_dct("input.jpeg")
>>> im.Y; im.Cb; im.Cr; im.qt

With **jpeglib** you can choose a particular version of *libjpeg* to
work with. Currently supported are all *libjpeg* versions from *6b* to *9e*,
and newest major and minor releases of *libjpeg-turbo* and *mozjpeg*.

>>> jpeglib.version.set('6b')
>>> im = jpeglib.read_spatial("input.jpeg")
>>> im.spatial

.. note::

   This project is under active development.

Contents
--------

.. toctree::
   :maxdepth: 2

   usage
   reference
   glossary
   faq