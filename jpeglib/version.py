
class version:
    """Class grouping functions for controlling libjpeg method."""
    @staticmethod
    def set(version):
        """Sets the version of libjpeg to use. Loads the library.
        
        :param version: Version to use, one of 6b, 8d.
        :type version: str
        :raises [NotImplementedError]: When unsupported libjpeg version is passed.
        """
        from ._bind import CJpegLib
        if version in {'6','6b'}:
            CJpegLib.set_version(version='6b')
        elif version in {'8','8d'}:
            CJpegLib.set_version(version='8d')
        else:
            raise NotImplementedError(f'Unsupported libjpeg version')
    @staticmethod
    def get():
        """Gets the currently used version of libjpeg. 
        
        :return: libjpeg version or None if not been loaded yet.
        :rtype: str, None
        """
        from ._bind import CJpegLib
        return CJpegLib.get_version()
    @staticmethod
    def get_lib():
        """Low-level getter of the dynamic library.
        
        :return: Dynamic library object or None if not loaded yet.
        :rtype: ctypes.CDLL, None
        """
        from ._bind import CJpegLib
        return CJpegLib.get()

__all__ = ['version']