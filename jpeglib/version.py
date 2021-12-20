
from ._bind import CJpegLib

class version:
    """Class grouping functions for controlling libjpeg method."""
    @staticmethod
    def set(version):
        """Sets the version of libjpeg to use. Loads the library.
        
        :param version: libjpeg version, one of 6b, 8d, 9d, turbo210.
        :type version: str
        :raises [NotImplementedError]: unsupported libjpeg version

        :Example:

        >>> import jpeglib
        >>> jpeglib.version.set('8d')
        """
        if version in {'6','6b'}:
            CJpegLib.set_version(version='6b')
        elif version in {'8','8d'}:
            CJpegLib.set_version(version='8d')
        elif version in {'9','9d'}:
            CJpegLib.set_version(version='9d')
        elif version in {'turbo2.1.0','turbo2.1','turbo210','turbo21'}:
            CJpegLib.set_version(version='turbo210')
        else:
            raise NotImplementedError(f'Unsupported libjpeg version')
    @staticmethod
    def get():
        """Gets the currently used version of libjpeg. 
        
        :return: libjpeg version or None if not been loaded yet.
        :rtype: str, None


        :Example:

        >>> import jpeglib
        >>> jpeglib.version.set('6b')
        >>> jpeglib.version.get()
        '6b'
        """
        return CJpegLib.get_version()
    @staticmethod
    def _jpeg_lib_version():
        """Returns value of jpeg_lib_version macro."""
        return CJpegLib.jpeg_lib_version()
    @staticmethod
    def _get_lib():
        """Low-level getter of the dynamic library.
        
        :return: Dynamic library object or None if not loaded yet.
        :rtype: ctypes.CDLL, None
        """
        return CJpegLib.get()
    @staticmethod
    def versions():
        """Lists present DLLs of versions."""
        return CJpegLib.versions()

__all__ = ['version']