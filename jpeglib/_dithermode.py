
from dataclasses import dataclass,field
from ._cstruct import CStruct

@dataclass
class Dithermode(CStruct):
    
    _J_DITHER_MODE = {
        "JDITHER_NONE":     0, #
        "JDITHER_ORDERED":  1, #
        "JDITHER_FS":       2, #
    }
    @classmethod
    def J_DITHER_MODE(cls):
        return cls._J_DITHER_MODE
    @classmethod
    def iJ_DITHER_MODE(cls):
        return {v: k for k, v in cls._J_DITHER_MODE.items()}
    
    @classmethod
    def name_to_index(cls, name):
        return cls.J_DITHER_MODE()[name]
    @classmethod
    def index_to_name(cls, index):
        return cls.iJ_DITHER_MODE()[index]

    def __repr__(self):
        return '<Dithermode %s>' % self.name