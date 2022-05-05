#!/usr/bin/env python3
"""Module with color space operations."""

from dataclasses import dataclass,field
from ._cstruct import CStruct

@dataclass
class Colorspace(CStruct):
    """Representation of color space.
    
    Allows mapping name-index and acquiral of number of channels.
    """

    _J_COLOR_SPACE = {
        "JCS_UNKNOWN":      0, # Unspecified color space
        "JCS_GRAYSCALE":    1, # Monochrome
        "JCS_RGB":          2, # Standard RGB
        "JCS_YCbCr":        3, # YCbCr or YUB, standard YCC
        "JCS_CMYK":         4, # CMYK
        "JCS_YCCK":         5, # YCbCrK
    }
    """List of color spaces. Taken from libjpeg internals."""
    
    @classmethod
    def J_COLOR_SPACE(cls) -> dict:
        """Get color space name to index mapping.
        
        :return: Dictionary, mapping color space name to index.
        :rtype: dict
        
        :Example:
        
        >>> name2index = Colorspace.J_COLOR_SPACE()
        >>> name2index['JCS_RGB'] # -> 2
        """
        return cls._J_COLOR_SPACE
    @classmethod
    def iJ_COLOR_SPACE(cls) -> dict:
        """Get index to color space name mapping.
        
        :return: Dictionary, mapping index to color space name.
        :rtype: dict
        
        :Example:
        
        >>> index2name = Colorspace.iJ_COLOR_SPACE()
        >>> index2name[1] # -> 'JCS_GRAYSCALE'
        """
        return {v: k for k, v in cls._J_COLOR_SPACE.items()}
    @classmethod
    def name_to_index(cls, name:str) -> int:
        """Map color space name onto index.
        
        :param name: Color space name.
        :type name: str
        :return: Index
        :rtype: int
        
        :Example:
        
        >>> Colorspace.name_to_index('JCS_GRAYSCALE') # -> 1
        >>> Colorspace.name_to_index('JCS_RGB') # -> 2
        """
        return cls.J_COLOR_SPACE()[name]
    @classmethod
    def index_to_name(cls, index:int):
        """Map index onto color space name.
        
        :param index: Index.
        :type name: int
        :return: Color space name
        :rtype: str
        
        :Example:
        
        >>> Colorspace.index_to_name(1) # -> 'JCS_GRAYSCALE')
        >>> Colorspace.index_to_name(2) # -> 'JCS_RGB')
        """
        return cls.iJ_COLOR_SPACE()[index]
    
    @property
    def channels(self) -> int:
        """Get channels of the colorspace.
        
        :return: Number of channels
        :rtype: int
        
        :Example:
        
        >>> colorspace = Colorspace("JCS_RGB")
        >>> colorspace # -> 3
        
        >>> colorspace = Colorspace("JCS_GRAYSCALE")
        >>> colorspace # -> 1
        """
        channel_no = {
            "JCS_GRAYSCALE": 1,
            "JCS_RGB":       3,
            "JCS_YCbCr":     3,
            "JCS_CMYK":      4,
            "JCS_YCCK":      4,
        }
        if self.name == "JCS_UNKNOWN":
            raise Exception("can't get number of channels for JCS_UNKNOWN colorspace")
        return channel_no[self.name]
    
    def __repr__(self):
        return '<Colorspace %s>' % self.name
    
    @classmethod
    def from_index(cls, index:int):
        """Construct from index.
        
        :param index: Color space index.
        :type index: int
        
        :Example:
        
        >>> colorspace = Colorspace.from_index(2)
        >>> colorspace.name # -> "JCS_RGB"
        """
        name = cls.iJ_COLOR_SPACE()[index]
        return cls(name=name)
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, int):
            return self.index == other
        else:
            return self.index == other.index