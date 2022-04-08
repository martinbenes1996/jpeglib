
import numpy as np

from ._base import *

def _dct_ii(X:np.ndarray, u:int, v:int):
    """DCT coefficient at index u,v.
    
    :param X: input pixel data tensor 
    :type X: np.ndarray
    :param u: horizontal index of DCT coefficient
    :type u: int
    :param v: vertical index of DCT coefficient
    :type v: int
    :return: output DCT coefficient value
    :rtype: float
    
    :Examples:
    
    >>> X = np.array([[1,2],[3,1]])
    >>> _dct_ii(X,1,0) # -> 2.72e-16
    """
    N,M = X.shape
    Yuv = (
        np.sqrt(2/N) * np.sqrt(2/M) *
        Lambda(u) * Lambda(v) *
        sum(
            X[i,j] * np.cos(np.pi/N*(i+.5)*u) * np.cos(np.pi/N*(j+.5)*v)
        for i in range(N)
        for j in range(M))
    )
    return Yuv

def _dct_iii(Y, i, j):
    """Pixel value at index i,j.
    
    :param x: input DCT coefficient tensor
    :type x: np.ndarray
    :param i: horizontal index of the pixel
    :type i: int
    :param j: vertical index of the pixel
    :type j: int
    :return: output value
    :rtype: float
    """
    N,M = Y.shape
    Xij = (
        sqrt(2/N) * sqrt(2/M) *
        sum(
            Lambda(u) * Lambda(v) * Y[u,v] * np.cos(np.pi/N*(i+.5)*u) * np.cos(np.pi/N*(j+.5)*v)
        for u in range(N)
        for v in range(M))
    )
    return Xij

def DCT2D(X:np.ndarray):
    """Python implementation of 2D DCT transformation.
    
    :param X: pixel data in spatial domain
    :type X: np.ndarray
    :return: tensor of DCT coefficients
    :rtype: np.ndarray
    
    :Examples:
    
    >>> X = np.array([[1,2],[3,1]])
    >>> Y = jpeglib.dct.DCT2D(X)
    """
    return np.array([[_dct_ii(X, u,v) for v in range(X.shape[1])] for u in range(X.shape[0])])

def iDCT2D(Y:np.ndarray):
    """Python implementation of 2D inverse DCT transformation.
    
    :param Y: tensor of DCT coefficients
    :type Y: np.ndarray
    :return: pixel data in spatial domain
    :rtype: np.ndarray
    
    :Examples:
    
    >>> X = np.array([[1,2],[3,1]])
    >>> Y = jpeglib.dct.DCT2D(X)
    >>> jpeglib.dct.iDCT2d(Y) # -> reconstruction of X
    """
    return np.array([[_dct_iii(Y, i,j) for j in range(Y.shape[1])] for i in range(Y.shape[0])])

__all__ = ["DCT2D", "iDCT2D"]