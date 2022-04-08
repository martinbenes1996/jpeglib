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
   :members: Y, Cb, Cr, qt, write_dct

Inside spatial domain
"""""""""""""""""""""

.. autofunction:: jpeglib.read_spatial

.. autofunction:: jpeglib.from_spatial

.. autoclass:: jpeglib.SpatialJPEG
   :members: spatial, write_spatial, color_space, dither_mode, dct_method, flags

Using jpegio interface
""""""""""""""""""""""

.. autofunction:: jpeglib.to_jpegio

.. autoclass:: jpeglib.DCTJPEGio
   :members: coef_arrays, quant_tables, write

libjpeg-like structures
"""""""""""""""""""""""

.. autoclass:: jpeglib.Dithermode
   :members: from_index, name, index, name_to_index, index_to_name, J_DITHER_MODE, iJ_DITHER_MODE

.. autoclass:: jpeglib.Colorspace
   :members: from_index, name, index, name_to_index, index_to_name, J_COLOR_SPACE, iJ_COLOR_SPACE

.. autoclass:: jpeglib.DCTMethod
   :members: from_index, name, index, name_to_index, index_to_name, J_DCT_METHOD, iJ_DCT_METHOD

.. autoclass:: jpeglib.Marker
   :members: from_index, name, index, name_to_index, index_to_name, J_MARKER_CODE, iJ_MARKER_CODE



Manage libjpeg version
----------------------

.. autoclass:: jpeglib.version
   :members: set, get, versions
   :special-members: __enter__, __exit__


DCT implementation
------------------

DCT implementation in 2 dimensions
""""""""""""""""""""""""""""""""""

.. autofunction:: jpeglib.dct.DCT2D

.. autofunction:: jpeglib.dct.iDCT2D


DCT implementation in 1 dimensions
""""""""""""""""""""""""""""""""""

.. autofunction:: jpeglib.dct.DCT1D

.. autofunction:: jpeglib.dct.iDCT1D
