"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

from dataclasses import dataclass
from ._cenum import MarkerType


@dataclass
class Marker:
    """Representation of JPEG marker."""

    type: MarkerType
    """type of marker"""
    length: int
    """length (in bytes)"""
    content: bytes
    """content of the marker"""

    def __repr__(self) -> str:
        data_size = 0
        if self.content is not None:
            data_size = len(self.content)
        return '<Marker %s N=%s>' % (str(self.type), data_size)

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
        """Marker content."""
        return self._content

    @content.setter
    def content(self, content: bytes):
        self._content = content
