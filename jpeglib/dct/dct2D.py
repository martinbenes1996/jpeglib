
from numba import jit, prange
import numpy as np

from ._common import Lambda

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _dct_ii(X: np.ndarray, Y: np.ndarray, N:int, M:int) -> float:
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
    >>> Y = np.zeros((2, 2))
    >>> _dct_ii(X, Y) # ->
    """
    for u in prange(N):
        for v in prange(M):
            Y[u][v] = (
                np.sqrt(2/N) *
                np.sqrt(2/M) *
                Lambda(u) *
                Lambda(v)
            )
            sum_value:float = 0
            for i in prange(N):
                for j in prange(M):
                    sum_value = sum_value + (
                        X[i][j] *
                        np.cos(np.pi/N*(i+.5)*u) *
                        np.cos(np.pi/N*(j+.5)*v)
                    )
            Y[u][v] = Y[u][v] * sum_value
    return Y

    # Yuv = (
    #     np.sqrt(2/N) *
    #     np.sqrt(2/M) *
    #     Lambda(u) *
    #     Lambda(v) *
    #     sum(
    #         X[i, j] *
    #         np.cos(np.pi/N*(i+.5)*u) *
    #         np.cos(np.pi/N*(j+.5)*v)
    #         for i in range(N) for j in range(M)
    #     )
    # )
    # return Yuv

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _dct_iii(Y: np.ndarray, X: np.ndarray, N:int, M:int) -> float:
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
    for i in prange(N):
        for j in prange(M):
            X[i, j] = (
                np.sqrt(2/N) *
                np.sqrt(2/M)
            )
            sum_value:float = 0
            for u in prange(N):
                for v in prange(M):
                    sum_value = sum_value + (
                        Lambda(u) *
                        Lambda(v) *
                        Y[u][v] *
                        np.cos(np.pi/N*(i + .5)*u) *
                        np.cos(np.pi/N*(j + .5)*v)
                    )
            X[i][j] = X[i][j] * sum_value
    return X

    # N, M = Y.shape
    # Xij = (
    #     np.sqrt(2/N) *
    #     np.sqrt(2/M) *
    #     sum(
    #         (
    #             Lambda(u) *
    #             Lambda(v) *
    #             Y[u, v] *
    #             np.cos(np.pi/N*(i + .5)*u) *
    #             np.cos(np.pi/N*(j + .5)*v)
    #         )
    #         for u in range(N) for v in range(M)
    #     )
    # )
    # return Xij


# @jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
# def _DCT2D(X: np.ndarray, Y:np.ndarray, N:int, M:int) -> np.ndarray:
#     for u in prange(N):
#         for v in prange(M):
#             Y[u][v] = _dct_ii(X[8*u:8*u+8][8*v:8*v+8], Y[u, v], N, M)
#     return Y

# @jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
# def _iDCT2D(Y:np.ndarray, X: np.ndarray, N:int, M:int) -> np.ndarray:
#     for u in prange(N):
#         for v in prange(M):
#             X[8*u:8*u+8][8*v:8*v+8] = _dct_iii(Y[u][v], X[8*u:8*u+8][8*v:8*v+8], N, M)
#     return X



def DCT2D(X: np.ndarray) -> np.ndarray:
    """Python implementation of 2D DCT transformation.

    :param X: pixel data in spatial domain
    :type X: np.ndarray
    :return: tensor of DCT coefficients
    :rtype: np.ndarray

    :Examples:

    >>> X = np.array([[1,2],[3,1]])
    >>> Y = jpeglib.dct.DCT2D(X)
    """
    N, M = X.shape
    Y = np.zeros(X.shape)
    Y = _dct_ii(X, Y, N, M)
    return Y
    # return np.array([
    #     [
    #         _dct_ii(X, u, v)
    #         for v in range(X.shape[1])
    #     ]
    #     for u in range(X.shape[0])
    # ])


def iDCT2D(Y: np.ndarray) -> np.ndarray:
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
    N, M = Y.shape
    X = np.zeros((N, M))
    X = _dct_iii(Y, X, N, M)
    return X
    # return np.array([
    #     [
    #         _dct_iii(Y, i, j)
    #         for j in range(Y.shape[1])
    #     ]
    #     for i in range(Y.shape[0])
    # ])

# def DCT2D(X: np.ndarray) -> np.ndarray:
#     """Python implementation of 2D DCT transformation.

#     :param X: pixel data in spatial domain
#     :type X: np.ndarray
#     :return: tensor of DCT coefficients
#     :rtype: np.ndarray

#     :Examples:

#     >>> X = np.array([[1,2],[3,1]])
#     >>> Y = jpeglib.dct.DCT2D(X)
#     """
#     N, M = [int(np.ceil(i/8)) for i in X.shape]
#     Y = np.zero((N, M, 8, 8))
#     Y = _DCT2D(X, Y, N, M)
#     return Y

# def iDCT2D(Y:np.ndarray) -> np.ndarray:
#     """Python implementation of 2D DCT transformation.

#     :param X: pixel data in spatial domain
#     :type X: np.ndarray
#     :return: tensor of DCT coefficients
#     :rtype: np.ndarray

#     :Examples:

#     >>> X = np.array([[1,2],[3,1]])
#     >>> Y = jpeglib.dct.DCT2D(X)
#     """
#     N, M, _, _ = Y.shape
#     X = np.zeros((N*8, M*8))
#     X = _iDCT2D(Y, X, N, M)
#     return X



__all__ = ["DCT2D", "iDCT2D"]
# __all__ = ["DCT2D", "iDCT2D", "block_DCT2D", "block_iDCT2D"]
