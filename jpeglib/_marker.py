
from dataclasses import dataclass,field
from ._cstruct import CStruct

@dataclass
class Marker(CStruct):
    
    _J_MARKER_CODE = {
        "JPEG_RST0":    0xD0, #
        "JPEG_RST1":    0xD0+1, #
        "JPEG_RST2":    0xD0+2, #
        "JPEG_RST3":    0xD0+3, #
        "JPEG_RST4":    0xD0+4, #
        "JPEG_RST5":    0xD0+5, #
        "JPEG_RST6":    0xD0+6, #
        "JPEG_RST7":    0xD0+7, #
        "JPEG_RST8":    0xD0+8, #
        "JPEG_EOI":     0xD9, #
        "JPEG_APP0":    0xE0, #
        "JPEG_APP1":    0xE0+1, #
        "JPEG_APP2":    0xE0+2, #
        "JPEG_APP3":    0xE0+3, #
        "JPEG_APP4":    0xE0+4, #
        "JPEG_APP5":    0xE0+5, #
        "JPEG_APP6":    0xE0+6, #
        "JPEG_APP7":    0xE0+7, #
        "JPEG_APP8":    0xE0+8, #
        "JPEG_APP9":    0xE0+9, #
        "JPEG_APP10":   0xE0+10, #
        "JPEG_APP11":   0xE0+11, #
        "JPEG_APP12":   0xE0+12, #
        "JPEG_APP13":   0xE0+13, #
        "JPEG_APP14":   0xE0+14, #
        "JPEG_APP15":   0xE0+15, #
        "JPEG_COM":     0xFE,
    }
    length:int
    content:bytes
    
    @classmethod
    def J_MARKER_CODE(cls):
        return cls._J_MARKER_CODE
    @classmethod
    def iJ_MARKER_CODE(cls):
        return {v: k for k, v in cls._J_MARKER_CODE.items()}
    
    @classmethod
    def name_to_index(cls, name):
        return cls.J_MARKER_CODE()[name]
    @classmethod
    def index_to_name(cls, index):
        return cls.iJ_MARKER_CODE()[index]

    def __repr__(self):
        data_size = 0
        if self.content is not None:
            data_size = len(self.content)
        return '<Marker %s N=%s>' % (self.name, data_size)
    def __len__(self):
        return self.length

    @property
    def length(self) -> int:
        return self._length
    @length.setter
    def length(self, length: int):
        self._length = length
        
    @property
    def content(self) -> bytes:
        return self._content
    @content.setter
    def content(self, content: bytes):
        self._content = content
    
    @classmethod
    def from_index(cls, index:int, length:int, content:bytes):
        name = cls.index_to_name(index)
        return cls(name=name, length=length, content=content)