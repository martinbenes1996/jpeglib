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

libjpeg-like structures
"""""""""""""""""""""""

.. autoclass:: jpeglib.Dithermode
   :members: from_index, name, index, name_to_index, index_to_name, J_DITHER_MODE, iJ_DITHER_MODE

.. autoclass:: jpeglib.Colorspace
   :members: from_index, name, index, name_to_index, index_to_name, J_COLOR_SPACE, iJ_COLOR_SPACE, channels

.. autoclass:: jpeglib.DCTMethod
   :members: from_index, name, index, name_to_index, index_to_name, J_DCT_METHOD, iJ_DCT_METHOD

.. autoclass:: jpeglib.Marker
   :members: from_index, name, index, name_to_index, index_to_name, J_MARKER_CODE, iJ_MARKER_CODE, content, length



Manage libjpeg version
----------------------

.. autoclass:: jpeglib.version
   :members: set, get, versions
   :special-members: __enter__, __exit__


Miscellaneous operations with images
------------------------------------

DCT implementation
""""""""""""""""""

.. autofunction:: jpeglib.ops.forward_dct

.. autofunction:: jpeglib.ops.backward_dct


JPEG compression primitives
"""""""""""""""""""""""""""

.. autofunction:: jpeglib.ops.blockify_8x8

.. autofunction:: jpeglib.ops.grayscale
