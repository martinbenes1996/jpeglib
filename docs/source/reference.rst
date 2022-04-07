Reference
=========

.. contents:: Table of Contents
   :local:

Working with jpegs
------------------

.. autofunction:: jpeglib.read_dct

.. autofunction:: jpeglib.read_spatial

.. autoclass:: jpeglib.SpatialJPEG
   :members: spatial, write_spatial, color_space, dither_mode, dct_method, flags
   :special-members: __init__

.. autoclass:: jpeglib.DCTJPEG
   :members: Y, Cb, Cr, qt, write_dct
   :special-members: __init__

.. autoclass:: jpeglib.Dithermode
   :members: from_index, name, index, name_to_index, index_to_name, J_DITHER_MODE, iJ_DITHER_MODE
   :special-members: __init__

.. autoclass:: jpeglib.Colorspace
   :members: from_index, name, index, name_to_index, index_to_name, J_COLOR_SPACE, iJ_COLOR_SPACE
   :special-members: __init__

.. autoclass:: jpeglib.DCTMethod
   :members: from_index, name, index, name_to_index, index_to_name, J_DCT_METHOD, iJ_DCT_METHOD
   :special-members: __init__

.. autoclass:: jpeglib.Marker
   :members: from_index, name, index, name_to_index, index_to_name, J_MARKER_CODE, iJ_MARKER_CODE
   :special-members: __init__

.. autofunction:: jpeglib.to_jpegio

Manage libjpeg version
----------------------

.. autoclass:: jpeglib.version
   :members:
   :special-members: __enter__, __exit__
