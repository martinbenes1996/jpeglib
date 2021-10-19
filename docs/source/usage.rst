Usage
=====

.. contents:: Table of Contents

Setup
-----

To use ``jpeglib``, first install it using pip:

.. code-block:: console

   $ pip3 install jpeglib

Import the package with

>>> import jpeglib

You can specify a particular libjpeg to use with
:py:func:`set_libjpeg_version()`.

>>> jpeglib.set_libjpeg_version('6b')

Currently supported versions are ``"6b"`` and ``"8d"``. 

Pixel data
----------

Get pixel data

TODO

DCT coefficients
----------------

Get DCT coefficients and quantization matrix

TODO