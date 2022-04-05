
from dataclasses import dataclass,field
from ._cstruct import CStruct

@dataclass
class DCTMethod(CStruct):
    
    _J_DCT_METHOD = {
        "JDCT_ISLOW":   0, #
        "JDCT_IFAST":   1, #
        "JDCT_FLOAT":   2, #
    }
    @classmethod
    def J_DCT_METHOD(cls):
        return cls._J_DCT_METHOD
    @classmethod
    def iJ_DCT_METHOD(cls):
        return {v: k for k, v in cls._J_DCT_METHOD.items()}
    
    @classmethod
    def name_to_index(cls, name):
        return cls.J_DCT_METHOD()[name]
    @classmethod
    def index_to_name(cls, index):
        return cls.iJ_DCT_METHOD()[index]

    def __repr__(self):
        return '<DCTMethod %s>' % self.name