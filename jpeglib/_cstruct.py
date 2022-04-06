
from dataclasses import dataclass,field

@dataclass
class CStruct:
    name: str
    
    @classmethod
    def name_to_index(cls, name):
        raise NotImplementedError
    @classmethod
    def index_to_name(cls, index):
        raise NotImplementedError()    

    @property
    def index(self) -> int:
        return self.name_to_index(self.name)
        
    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, name: str):
        self._name = name
    
    def __repr__(self):
        return '<CStruct %s>' % self.name
    def __str__(self):
        return self.name
    def __int__(self):
        return self.index
    
    @classmethod
    def from_index(cls, index):
        name = cls.index_to_name(index)
        return cls(name=name)