"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

import numpy as np
from ._cenum import Colorspace


def jpeg_color_space(
    Cb: np.ndarray,
    Cr: np.ndarray,
    K: np.ndarray
) -> Colorspace:
    """"""
    if Cb is None and Cr is None:  # grayscale
        colorspace = Colorspace.JCS_GRAYSCALE
    elif Cb is None or Cr is None:  # Cb or Cr not given
        raise IOError('both Cb and Cr must be non-zero')
    elif K is None:  # YCbCr
        colorspace = Colorspace.JCS_YCbCr
    else:  # K given
        colorspace = Colorspace.JCS_YCCK
    #
    return colorspace


def in_color_space(
    num_components: int
) -> Colorspace:
    """Inferes in_color_space from number of components."""
    if num_components == 3:
        colorspace = Colorspace.JCS_RGB
    elif num_components == 1:
        colorspace = Colorspace.JCS_GRAYSCALE
    else:
        raise IOError('failed to infere colorspace')
    return colorspace


IN_JPEG_COLORSPACE = {
    'JCS_GRAYSCALE': Colorspace.JCS_GRAYSCALE,
    'JCS_CMYK': Colorspace.JCS_YCCK,
    'JCS_RGB': Colorspace.JCS_YCbCr,
    'JCS_YCbCr': Colorspace.JCS_YCbCr,
}

def jpeg_in_color_space(
    in_color_space: Colorspace,
) -> Colorspace:
    """Returns jpeg_color_space corresponding to in_color_space."""
    # return IN_JPEG_COLORSPACE[str(in_color_space)]
    return IN_JPEG_COLORSPACE.get(
        str(in_color_space),
        Colorspace.JCS_YCbCr
    )

ASSIGNMENT_YCC = {
    1: np.array([0]),
    2: np.array([0, 1, 1]),
    3: np.array([0, 1, 2]),
    4: np.array([0, 1, 2, 3]),
}
ASSIGNMENT_YCCK = {
    2: np.array([0, 1, 1, 1]),
    3: np.array([0, 1, 1, 2]),
    4: np.array([0, 1, 2, 3]),
}

def quant_tbl_no(
    qt: np.ndarray,
    Cb: np.ndarray = None,
    Cr: np.ndarray = None,
    K: np.ndarray = None,
) -> np.ndarray:
    """Inferes quant_tbl_no to given non-luminance components.

    Either give Cb, Cr, K, or spatial.
    """
    if qt is None:  # no qt
        tbl_no = None
    # elif spatial is not None and qt.shape[0] in ASSIGNMENT_YCC:
    #     tbl_no = ASSIGNMENT_YCC[qt.shape[0]]
    # elif Cb is None and Cr is None:  # grayscale
    #     tbl_no = np.array([0])
    elif K is not None and qt.shape[0] in ASSIGNMENT_YCCK:
        tbl_no = ASSIGNMENT_YCCK[qt.shape[0]]
    elif qt.shape[0] in ASSIGNMENT_YCC:
        tbl_no = ASSIGNMENT_YCC[qt.shape[0]]
    else:
        raise Exception('failed to infere quant_tbl_no')
    #
    return tbl_no


def samp_factor(
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    if Cb is None and Cr is None:  # grayscale
        factor = np.array([1, 1])
    else:
        # collect dimensions
        dims = [
            i.shape[:2]
            for i in [Cb, Cr, K]
            if i is not None
        ]
        # compute sampling factor
        dims = np.array([np.array(Y.shape[:2]) / np.array(i) for i in dims])
        max_subs = np.max(dims, axis=0)
        factor = np.array([
            max_subs,
            *(max_subs / dims)
        ]).astype('int16')
    return factor


def block_dims(
    Y: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    return np.array([
        [i.shape[0], i.shape[1]]
        for i in [Y, Cb, Cr, K]
        if i is not None
    ])
