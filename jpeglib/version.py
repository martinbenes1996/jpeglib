"""
Module for operation with libjpeg versions.
"""

def set_libjpeg_version(version):
    """Sets the version of libjpeg to use.
    
    :param version: Version to use, one of 6b, 8d.
    :type version: str
    :raises [NotImplementedError]: When unsupported libjpeg version is passed.
    """
    from ._bind import CJpegLib
    if version in {'6','6b'}:
        CJpegLib._bind_lib(version='6b')
    elif version in {'8','8d'}:
        CJpegLib._bind_lib(version='8d')
    else:
        raise NotImplementedError(f'Unsupported libjpeg version')
