
import numpy as np


def blockify_8x8(x: np.ndarray) -> np.ndarray:
    """Splits the spatial domain into 8x8 tiles.

    The output has shape (N/8, M/8, 8, 8, channels).

    :param x: input pixel array
    :type x: np.ndarray
    :return: tiled pixels
    :rtype: np.ndarray

    :Examples:

    Example of usage with jpeglib

    >>> import jpeglib
    >>> im = jpeglib.read_spatial('image.jpeg')
    >>> r = jpeglib.ops.blockify_8x8(im.x)
    """
    # get shape
    N, M, _ = x.shape
    # split along each axis (separately)
    xh = np.split(x, (N+7)//8, axis=0)
    xhv = [np.split(r, (M+7)//8, axis=1) for r in xh]
    # merge
    xb = np.array(xhv)
    return xb

def deblockify_8x8(xb:np.ndarray) -> np.ndarray:
    """Merges split spatial domain (8x8 blocks).

    The output has shape (N, M) for input (N/8, M/8, 8, 8)

    :param xb: tiled pixels
    :type xb: np.ndarray
    :return: spatial domain representation
    :rtype: np.ndarray

    :Examples:

    Example of usage with jpeglib

    >>> import jpeglib
    >>> im = jpeglib.read_dct('image.jpeg')
    >>> X = jpeglib.backward_dct(im.Y)
    >>> r = jpeglib.ops.deblockify_8x8(X)
    """
    return (
        np.einsum('abcd->adbc', xb)
            .reshape(xb.shape[0]*8, xb.shape[1]*8)
    )


def decompress_lossless(Y:np.ndarray)

def luminance(x: np.ndarray) -> np.ndarray:
    """Converts RGB into luminance / grayscale.

    :param x: input rgb pixel array
    :type x: np.ndarray
    :return: grayscale pixel array
    :rtype: np.ndarray

    :Examples:

    Example of usage with jpeglib

    >>> import jpeglib
    >>> im = jpeglib.read_spatial('image.jpeg')
    >>> g = jpeglib.ops.luminance(im.x)
    """
    lumo_contributions = np.array([0.2989, 0.5870, 0.1140])
    return np.expand_dims(x @ lumo_contributions, 2)


def grayscale(x: np.ndarray) -> np.ndarray:
    return luminance(x=x)
