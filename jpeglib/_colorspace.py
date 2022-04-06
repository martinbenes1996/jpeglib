
from dataclasses import dataclass,field
from ._cstruct import CStruct
@dataclass
class Colorspace(CStruct):

    _J_COLOR_SPACE = {
        "JCS_UNKNOWN":      0, # Unspecified color space
        "JCS_GRAYSCALE":    1, # Monochrome
        "JCS_RGB":          2, # Standard RGB
        "JCS_YCbCr":        3, # YCbCr or YUB, standard YCC
        "JCS_CMYK":         4, # CMYK
        "JCS_YCCK":         5, # YCbCrK
    }
    
    @classmethod
    def J_COLOR_SPACE(cls):
        return cls._J_COLOR_SPACE
    @classmethod
    def iJ_COLOR_SPACE(cls):
        return {v: k for k, v in cls._J_COLOR_SPACE.items()}
    @classmethod
    def name_to_index(cls, name):
        return cls.J_COLOR_SPACE()[name]
    @classmethod
    def index_to_name(cls, index):
        return cls.iJ_COLOR_SPACE()[index]
    
    @property
    def channels(self):
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
    def from_index(cls, index):
        name = cls.iJ_COLOR_SPACE()[index]
        return cls(name=name)