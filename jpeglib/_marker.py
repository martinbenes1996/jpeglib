"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

from dataclasses import dataclass
from ._cenum import MarkerType


@dataclass
class Marker:

    type: MarkerType
    length: int
    content: bytes

    def __repr__(self) -> str:
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
        """Marker content."""
        return self._content

    @content.setter
    def content(self, content: bytes):
        self._content = content
