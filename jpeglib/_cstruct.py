
from dataclasses import dataclass,field

@dataclass
class CStruct:
    name: str
    """item name"""
    
    @classmethod
    def name_to_index(cls, name):
        raise NotImplementedError
    @classmethod
    def index_to_name(cls, index):
        raise NotImplementedError()    

    @property
    def index(self) -> int:
        """Index getter."""
        return self.name_to_index(self.name)
        
    @property
    def name(self) -> str:
        """Name getter."""
        return self._name
    @name.setter
    def name(self, name: str):
        """Name setter."""
        self._name = name
    
    def __repr__(self) -> str:
        return '<CStruct %s>' % self.name
    def __str__(self) -> str:
        """Converts the class to str, returns name."""
        return self.name
    def __int__(self) -> int:
        """Converts the class to int, returns index."""
        return self.index
    
    @classmethod
    def from_index(cls, index):
        """Constructs the class from index. Uses the reverse mapping onto name."""
        name = cls.index_to_name(index)
        return cls(name=name)