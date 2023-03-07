Reference
=========

.. contents:: Table of Contents
   :local:

Working with jpegs
------------------

Inside DCT domain
"""""""""""""""""

.. autofunction:: jpeglib.read_dct

.. autoclass:: jpeglib.DCTJPEG
   :members: path, Y, Cb, Cr, qt, write_dct, content, height, width, height_in_blocks, width_in_blocks, num_components, num_markers, samp_factor, progressive_mode, has_chrominance, quant_tbl_no, get_component_qt


.. autofunction:: jpeglib.from_dct

Inside spatial domain
"""""""""""""""""""""

.. autofunction:: jpeglib.read_spatial

.. autofunction:: jpeglib.from_spatial

.. autoclass:: jpeglib.SpatialJPEG
   :members: path, spatial, write_spatial, color_space, dither_mode, dct_method, flags, content, height, width, height_in_blocks, width_in_blocks, num_components, num_markers, samp_factor, progressive_mode, has_chrominance

Using jpegio interface
""""""""""""""""""""""

.. autofunction:: jpeglib.to_jpegio

.. autoclass:: jpeglib.DCTJPEGio
   :members: coef_arrays, quant_tables, write

libjpeg-like enumerations
"""""""""""""""""""""""""

.. autoclass:: jpeglib.Dithermode
   :members: JDITHER_NONE, JDITHER_ORDERED, JDITHER_FS

.. autoclass:: jpeglib.Colorspace
   :members: JCS_UNKNOWN, JCS_GRAYSCALE, JCS_RGB, JCS_YCbCr, JCS_CMYK, JCS_YCCK

.. autoclass:: jpeglib.DCTMethod
   :members: JDCT_ISLOW, JDCT_IFAST, JDCT_FLOAT

.. autoclass:: jpeglib.Marker
   :members: JPEG_RST0, JPEG_RST1, JPEG_RST2, JPEG_RST3, JPEG_RST4, JPEG_RST5, JPEG_RST6, JPEG_RST7, JPEG_RST8, JPEG_EOI, JPEG_APP0, JPEG_APP1, JPEG_APP2, JPEG_APP3, JPEG_APP4, JPEG_APP5, JPEG_APP6, JPEG_APP7, JPEG_APP8, JPEG_APP9, JPEG_APP10, JPEG_APP11, JPEG_APP12, JPEG_APP13, JPEG_APP14, JPEG_APP15, JPEG_COM



Manage libjpeg version
----------------------

.. autoclass:: jpeglib.version
   :members: set, get, versions
   :special-members: __enter__, __exit__

