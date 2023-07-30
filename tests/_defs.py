
import numpy as np
from parameterized import parameterized

ALL_VERSIONS = [
    ['6b'],
    ['7'],
    ['8'],
    ['8a'],
    ['8b'],
    ['8c'],
    ['8d'],
    ['9'],
    ['9a'],
    ['9b'],
    ['9c'],
    ['9d'],
    ['9e'],
    ['turbo120'],
    ['turbo130'],
    ['turbo140'],
    ['turbo150'],
    ['turbo200'],
    ['turbo210'],
    # ['turbo300'],
    ['mozjpeg101'],
    ['mozjpeg201'],
    ['mozjpeg300'],
    ['mozjpeg403'],
]


LIBJPEG_VERSIONS = [
    ['6b'],
    ['7'],
    ['8'],
    ['8a'],
    ['8b'],
    ['8c'],
    ['8d'],
    ['9'],
    ['9a'],
    ['9b'],
    ['9c'],
    ['9d'],
    ['9e'],
]


def version_cluster(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


# https://www.sciencedirect.com/topics/computer-science/quantization-matrix
qt50_standard = np.array([
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]],
    [[17, 18, 24, 47, 99, 99, 99, 99],
     [18, 21, 26, 66, 99, 99, 99, 99],
     [24, 26, 56, 99, 99, 99, 99, 99],
     [47, 66, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99],
     [99, 99, 99, 99, 99, 99, 99, 99]],
])
