

from __future__ import annotations
import dataclasses
import numpy as np

@dataclasses.dataclass
class Huffman:
    bits: np.ndarray
    """bits used to represent number of elements"""
    values: np.ndarray
    """values, ordered by number of bits represented (i.e., histogram)"""

    # def __post_init__(self):
    #     print(self.bits)
    #     print(len(np.unique(self.values)))
    #     print(self.values)
    #     assert len(self.values) == np.sum(self.bits), 'invalid histogram'

    @property
    def bits(self) -> str:
        """Name getter."""
        return self._bits

    @bits.setter
    def bits(self, bits: str):
        """Bits setter."""
        self._bits = bits

    @property
    def values(self) -> str:
        """Values getter."""
        return self._values

    @values.setter
    def values(self, values: str):
        """Values setter."""
        self._values = values

    def __repr__(self) -> str:
        bits = ','.join(map(str, self.bits[:5]))
        return f'<Huffman {bits},...>'

    def __str__(self) -> str:
        """Converts the class to str, returns name."""
        return repr(self)

    def __eq__(self, other) -> bool:
        """Compares two huffman tables for equality."""
        return (self.bits == other.bits).all() and (self.values == other.values).all()
