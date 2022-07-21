
from dataclasses import dataclass
from ._cstruct import CStruct


@dataclass
class DCTMethod(CStruct):

    _J_DCT_METHOD = {
        "JDCT_ISLOW":   0,  #
        "JDCT_IFAST":   1,  #
        "JDCT_FLOAT":   2,  #
    }

    @classmethod
    def J_DCT_METHOD(cls) -> dict:
        return cls._J_DCT_METHOD

    @classmethod
    def iJ_DCT_METHOD(cls) -> dict:
        return {v: k for k, v in cls._J_DCT_METHOD.items()}

    @classmethod
    def name_to_index(cls, name: str) -> str:
        return cls.J_DCT_METHOD()[name]

    @classmethod
    def index_to_name(cls, index: int) -> str:
        return cls.iJ_DCT_METHOD()[index]

    def __repr__(self) -> str:
        return '<DCTMethod %s>' % self.name
