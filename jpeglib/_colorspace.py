
from dataclasses import dataclass,field

@dataclass
class Colorspace:
    name: str
    
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
    
    @property
    def index(self) -> int:
        return self.J_COLOR_SPACE()[name]
        
    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, name: str):
        self._name = name
    
    def __repr__(self):
        return '<Colorspace %s>' % self.name
    def __str__(self):
        return self.name()
    def __int__(self):
        return self.index()
    
    @classmethod
    def from_index(cls, index):
        name = cls.iJ_COLOR_SPACE()[index]
        return cls(name=name)