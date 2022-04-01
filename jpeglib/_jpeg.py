
import ctypes
from collections import namedtuple
from dataclasses import dataclass
import numpy as np
from ._bind import CJpegLib
from ._colorspace import Colorspace

@dataclass
class JPEG:
    """JPEG abstract class."""
    path: str
    content: bytes
    height: int
    width: int
    block_dims: np.ndarray
    samp_factor: np.ndarray
    jpeg_color_space: Colorspace
    num_components: int
    
    def _free(self):
        raise NotImplementedError


def load_jpeg_info(path: str):
    # allocate
    _block_dims = (ctypes.c_int*6)()
    _image_dims = (ctypes.c_int*2)()
    _num_components = (ctypes.c_int*1)()
    _samp_factor = (ctypes.c_int*6)()
    _jpeg_color_space = (ctypes.c_int*1)()
    # call
    CJpegLib.read_jpeg_info(
        srcfile             = str(path),
        block_dims          = _block_dims,
        image_dims          = _image_dims,
        num_components      = _num_components,
        samp_factor         = _samp_factor,
        jpeg_color_space    = _jpeg_color_space
    )
    # process
    num_components = _num_components[0] # number of components in JPEG
    block_dims = (
        np.array([_block_dims[i] for i in range(2*num_components)], int)
        .reshape(num_components, 2)
    )
    samp_factor = (
        np.array([_samp_factor[i] for i in range(2*num_components)], int)
        .reshape(num_components, 2)
    )
    return JPEG(
        path                = path,
        height              = _image_dims[0],
        width               = _image_dims[1],
        block_dims          = block_dims,
        samp_factor         = samp_factor,
        num_components      = num_components,
        jpeg_color_space    = Colorspace.from_index(_jpeg_color_space[0]),
        content             = None
    )

    
def read_dct(path:str, jpeg:JPEG):
    # allocate DCT
    def _allocate_component(i:int, block_dims:np.ndarray):
        return (((ctypes.c_short * 64) * block_dims[i][1]) * block_dims[i][0])()
    _Y = _allocate_component(0, jpeg.block_dims)
    print("Allocated Y:", _Y)
    if jpeg.num_components > 1: # has chrominance
        _Cb = _allocate_component(1, jpeg.block_dims)
        print("Allocated Cb:", _Y)
        _Cr = _allocate_component(2, jpeg.block_dims)
        print("Allocated Cr:", _Y)
    else: # grayscale
        _Cb,_Cr = None,None
    # allocate QT
    _qt = ((ctypes.c_short * 64) * 4)()
    # call
    CJpegLib.read_jpeg_dct(
        srcfile     = path,
        Y           = _Y,
        Cb          = _Cb,
        Cr          = _Cr,
        qt          = _qt
    )
    # process
    qt = np.ctypeslib.as_array(_qt)
    qt = qt.reshape((*qt.shape[:-1],8,8))
    Y = np.ctypeslib.as_array(_Y)
    Y = Y.reshape((*Y.shape[:-1],8,8))
    if jpeg.num_components > 1: # has chrominance
        Cb = np.ctypeslib.as_array(_Cb)
        Cb = Cb.reshape((*Cb.shape[:-1],8,8))
        Cr = np.ctypeslib.as_array(_Cr)
        Cr = Cb.reshape((*Cr.shape[:-1],8,8))
    return Y,Cb,Cr,qt
    