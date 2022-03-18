
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
        elif version in {'7','8','8a','8b','8c','8d','9','9a','9b','9c','9d','9e'}:
            CJpegLib.set_version(version=version)
        elif version in {'turbo','turbo2.1.0','turbo2.1','turbo210','turbo21'}:
            CJpegLib.set_version(version='turbo210')
        elif version in {'mozjpeg4.0.3','mozjpeg4','mozjpeg403','mozjpeg'}:
            CJpegLib.set_version(version='mozjpeg403')
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
    def __init__(self, version):
        """Constructor, used in with statement.
        
        :param version: Version to set inside with block.
        :type version: str
        """
        self.next = version
    def __enter__(self):
        """Sets new version in a block.

        :Example:

        >>> jpeglib.version.set('6b')
        >>> # working with 6b
        >>> # [...]
        >>> with jpeglib.version('8d'):
        >>>     # working with 8d
        >>>     # [...]
        >>>     pass
        """
        self.prev = self.get()
        self.set(self.next)
    def __exit__(self, *args, **kw):
        """Recovers a previous version, when exiting `with` block.
        
        :Example:

        >>> # working with 6b
        >>> # [...]
        >>> with jpeglib.version('8d'):
        >>>     # working with 8d
        >>>     # [...]
        >>>     pass
        >>> # working with 6b (again)
        >>> # [...]
        """
        self.set(self.prev)

__all__ = ['version']