
from math import cos,pi
import numpy as np

from ._base import *

def _dct_ii(X:np.ndarray, u:int) -> float:
    """DCT coefficient at index u.
    
    :param X: input pixel data tensor 
    :type X: np.ndarray
    :param u: index of DCT coefficient
    :type u: int
    :return: output DCT coefficient value
    :rtype: float
    
    :Examples:
    
    >>> _dct_ii([1,2,1], 1) # -> 2.72e-16
    """
    N = len(x)
    Yu = np.sqrt(2/N) * Lambda(u) * sum(X[i] * np.cos(np.pi/N*(i+.5)*u) for i in range(N))
    return Yu

def _dct_iii(Y:np.ndarray, i:int) -> float:
    """Pixel value at index i.
    
    :param x: input DCT coefficient tensor
    :type x: np.ndarray
    :param i: index of the pixel
    :type i: int
    :return: output pixel value
    :rtype: float
    
    :Examples:
    
    >>> _dct_iii([2,1,-1], 1) # -> 1.97
    """
    N = len(x)
    Xi = np.sqrt(2/N) * sum(Lambda(u) * Y[u] * np.cos(np.pi/N*(i+.5)*u) for u in range(N))
    return Xi

def DCT1D(X:np.ndarray) -> np.ndarray:
    """Python implementation of 1D DCT transformation.
    
    :param X: pixel data in spatial domain
    :type X: np.ndarray
    :return: tensor of DCT coefficients
    :rtype: np.ndarray
    
    :Examples:
    
    >>> X = np.array([1,2,1])
    >>> Y = jpeglib.dct.DCT1D(X)
    """
    return [_dct_ii(X,k) for k in range(len(X))]

def iDCT1D(Y:np.ndarray):
    """Python implementation of 1D inverse DCT transformation.
    
    :param Y: tensor of DCT coefficients
    :type Y: np.ndarray
    :return: pixel data in spatial domain
    :rtype: np.ndarray
    
    :Examples:
    
    >>> X = np.array([1,2,1])
    >>> Y = jpeglib.dct.DCT1D(X)
    >>> jpeglib.dct.iDCT1D(Y) # -> reconstruction of X
    """
    return [_dct_iii(Y,k) for k in range(len(Y))]

__all__ = ["DCT1D", "iDCT1D"]