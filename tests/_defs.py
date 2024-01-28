"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from parameterized import parameterized

LIBJPEG_VERSIONS = [[v] for v in jpeglib.version.LIBJPEG_VERSIONS]

ALL_VERSIONS = [
    *LIBJPEG_VERSIONS,
    ["turbo120"],
    ["turbo130"],
    ["turbo140"],
    ["turbo150"],
    ["turbo200"],
    ["turbo210"],
    # ['turbo300'],
    ["mozjpeg101"],
    ["mozjpeg201"],
    ["mozjpeg300"],
    ["mozjpeg403"],
]

VERSIONS_EXCLUDE_MOZ = [
    *LIBJPEG_VERSIONS,
    ["turbo120"],
    ["turbo130"],
    ["turbo140"],
    ["turbo150"],
    ["turbo200"],
    ["turbo210"],
    # ['turbo300'],
]


def version_cluster(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


# https://www.sciencedirect.com/topics/computer-science/quantization-matrix
qt50_standard = np.array([
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ], [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
])

qt75_standard = np.array([
    [
        [8,   6,  5,  8, 12, 20, 26, 31],
        [6,   6,  7, 10, 13, 29, 30, 28],
        [7,   7,  8, 12, 20, 29, 35, 28],
        [7,   9, 11, 15, 26, 44, 40, 31],
        [9,  11, 19, 28, 34, 55, 52, 39],
        [12, 18, 28, 32, 41, 52, 57, 46],
        [25, 32, 39, 44, 52, 61, 60, 51],
        [36, 46, 48, 49, 56, 50, 52, 50]
    ], [
        [9,   9, 12, 24, 50, 50, 50, 50],
        [9,  11, 13, 33, 50, 50, 50, 50],
        [12, 13, 28, 50, 50, 50, 50, 50],
        [24, 33, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [50, 50, 50, 50, 50, 50, 50, 50]
    ]
])
