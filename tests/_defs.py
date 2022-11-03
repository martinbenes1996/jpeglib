
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

from parameterized import parameterized

def version_cluster(testcase_func, param_num, param):
    return "%s_%s" %(
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )