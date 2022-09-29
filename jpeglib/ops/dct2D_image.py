from numba import jit, prange
import numpy as np

@jit
def Lambda(x) -> np.ndarray:
    return 1 if x else 1/np.sqrt(2)

@jit
def precompute_cos() -> np.ndarray:
    cos_table = np.zeros((8,8))
    for u in prange(8):
        for i in prange(8):
            cos_table[u,i] = np.cos(np.pi/8*(i+.5)*u)
    return cos_table
@jit
def forward_dct(X:np.ndarray):
    # precompute cosines
    N, M, _, _ = X.shape
    cos_table = precompute_cos()

    # iterate blocks
    Y = np.zeros((N, M, 8, 8))
    for h in prange(N):
        for w in prange(M):

            # iterate coefficients
            for u in prange(8):
                for v in prange(8):

                    # iterate pixels
                    Y[h,w,u,v] = 0
                    for i in prange(8):
                        for j in prange(8):

                            # compute Y_uv
                            Y[h,w,u,v] += (
                                X[h,w,i,j] *
                                cos_table[u,i] *
                                cos_table[v,j])
                    Y[h,w,u,v] *= (
                        np.sqrt(2/8) *
                        np.sqrt(2/8) *
                        Lambda(u) *
                        Lambda(v))
    return Y

@jit
def backward_dct(Y:np.ndarray) -> np.ndarray:
    """Backward DCT on image coefficients.

    :param X: input DCT coefficients
    :type X: np.ndarray
    :return: pixels
    :rtype: np.ndarray

    :Examples:

    You can use this together with the other jpeglib structures.

    >>> import jpeglib
    >>> jpeg = jpeglib.read_dct("image.jpeg")
    >>> x_Y = jpeglib.dct.backward(jpeg.Y)

    """
    # precompute cosines
    N, M, _, _ = Y.shape
    cos_table = precompute_cos()

    # iterate blocks
    X = np.zeros((N, M, 8, 8))
    for h in prange(N):
        for w in prange(M):

            # iterate pixels
            for i in prange(8):
                for j in prange(8):

                    # iterate coefficients
                    X[h,w,i,j] = 0
                    for u in prange(8):
                        for v in prange(8):

                            # compute X_ij
                            X[h,w,i,j] += (
                                Lambda(u) *
                                Lambda(v) *
                                Y[h,w,u,v] *
                                cos_table[u,i] *
                                cos_table[v,j])
                    X[h,w,i,j] *= (
                        np.sqrt(2/8) *
                        np.sqrt(2/8))
    return X