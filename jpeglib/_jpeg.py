
import ctypes
from collections import namedtuple
from dataclasses import dataclass
import logging
import numpy as np
from ._bind import CJpegLib
from ._colorspace import Colorspace
from ._marker import Marker

@dataclass
class JPEG:
    """JPEG abstract class."""
    path: str
    """path to the jpeg file"""
    content: bytes
    """cached binary content of the jpeg"""
    height: int
    """image height"""
    width: int
    """image width"""
    block_dims: np.ndarray
    """DCT shapes in blocks; first is DCT component (0 Y,1 Cb,2 Cr), second is dimension (0 height, 1 width)"""
    samp_factor: np.ndarray
    """sampling factor; first is DCT component, (0 Y,1 Cb,2 Cr), second is orientation (0 horizontal, 1 vertical)"""
    jpeg_color_space: Colorspace
    """color space of the JPEG file"""
    num_components: int
    """number of components in the JPEG file"""
    markers: list
    """list of marker objects"""
    
    def height_in_blocks(self, component:int) -> int:
        """Getter of height in blocks.
        
        :param component: chroma component index (0 Y, 1 Cb, 2 Cr)
        :type component: int
        :return: chroma component height
        :rtype: int
        :raises [IndexError]: When component index is not one of 0,1,2
        """
        return self.block_dims[component][0]
    def width_in_blocks(self, component):
        return self.block_dims[component][1]
    def has_chrominance(self):
        return self.num_components > 1
    def num_markers(self) -> int:
        if self.markers is None:
            return 0
        return len(self.markers)
    
    def c_image_dims(self):
        return (ctypes.c_int * 2)(self.height,self.width)
    def c_block_dims(self):
        return (ctypes.c_int * 6)(
            self.block_dims[0][0], self.block_dims[0][1],
            self.block_dims[1][0], self.block_dims[1][1],
            self.block_dims[2][0], self.block_dims[2][1],)
    def c_samp_factor(self):
        if self.samp_factor is None:
            return self.samp_factor
        samp_factor = np.array(self.samp_factor, dtype=np.int32)
        return np.ctypeslib.as_ctypes(samp_factor)
    def c_marker_types(self):
        if self.markers is None:
            return None
        marker_types = [marker.index for marker in self.markers]
        return (ctypes.c_int32*self.num_markers())(*marker_types)
    def c_marker_lengths(self):
        if self.markers is None:
            return None
        marker_lengths = [marker.length for marker in self.markers]
        return (ctypes.c_int32*self.num_markers())(*marker_lengths)
    def c_markers(self):
        if self.markers is None:
            return None
        marker_lengths = [marker.length for marker in self.markers]
        marker_contents = []
        for marker in self.markers:
            marker_contents += [i for i in marker.content]
        return (ctypes.c_ubyte*np.sum(marker_lengths))(*marker_contents)
        
    def free(self):
        """Free the allocated tensors."""
        raise NotImplementedError


def load_jpeg_info(path: str):
    """"""
    # allocate
    _block_dims = (ctypes.c_int*6)()
    _image_dims = (ctypes.c_int*2)()
    _num_components = (ctypes.c_int*1)()
    _samp_factor = (ctypes.c_int*6)()
    _jpeg_color_space = (ctypes.c_int*1)()
    _marker_lengths = (ctypes.c_int*20)()
    _marker_types = (ctypes.c_uint32*20)()
    # call
    CJpegLib.read_jpeg_info(
        srcfile             = str(path),
        block_dims          = _block_dims,
        image_dims          = _image_dims,
        num_components      = _num_components,
        samp_factor         = _samp_factor,
        jpeg_color_space    = _jpeg_color_space,
        marker_lengths      = _marker_lengths,
        marker_types        = _marker_types,
    )
    # process
    num_components = _num_components[0] # number of components in JPEG
    block_dims = (
        np.array([_block_dims[i] for i in range(2*num_components)], int)
        .reshape(num_components, 2))
    samp_factor = (
        np.array([_samp_factor[i] for i in range(2*num_components)], int)
        .reshape(num_components, 2))
    markers = []
    for i in range(20):
        # marker length
        marker_length = _marker_lengths[i]
        if marker_length == 0:
            break
        # marker type
        marker_type = _marker_types[i]
        # create marker
        marker = Marker.from_index(
            index=marker_type, 
            length=marker_length, 
            content=None,
        )
        markers.append(marker)
    marker_lengths = np.array([marker.length for marker in markers], dtype=np.int32)
    num_markers = marker_lengths.shape[0]
    
    # allocate
    _markers = (ctypes.c_ubyte * np.sum(marker_lengths))()
    # call
    CJpegLib.read_jpeg_markers(
        srcfile             = str(path),
        markers             = _markers,
    )
    # process
    cumlens = np.cumsum([0] + marker_lengths.tolist())
    for i in range(num_markers):
        markers[i].content = bytes(_markers[cumlens[i]:cumlens[i+1]])
    # create jpeg
    return JPEG(
        path                = path,
        height              = _image_dims[0],
        width               = _image_dims[1],
        block_dims          = block_dims,
        samp_factor         = samp_factor,
        num_components      = num_components,
        jpeg_color_space    = Colorspace.from_index(_jpeg_color_space[0]),
        content             = None,
        markers             = markers,
    )
