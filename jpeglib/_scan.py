"""

Author: Martin Benes
Affiliation: Universitaet Innsbruck
"""

from __future__ import annotations
import dataclasses
import numpy as np


@dataclasses.dataclass
class Scan:
    """"""
    components: np.ndarray
    """index of components"""
    dc_tbl_no: np.ndarray
    """"""
    ac_tbl_no: np.ndarray
    """"""
    Ss: int
    """"""
    Se: int
    """"""
    Ah: int
    """"""
    Al: int
    """"""

    # def __post_init__(self):
    #     print(self.bits)
    #     print(len(np.unique(self.values)))
    #     print(self.values)
    #     assert len(self.values) == np.sum(self.bits), 'invalid histogram'

    @property
    def components(self) -> str:
        """Components getter."""
        return self._components

    @components.setter
    def components(self, components: str):
        """Components setter."""
        self._components = components

    @property
    def dc_tbl_no(self) -> str:
        """dc_tbl_no getter."""
        return self._dc_tbl_no

    @dc_tbl_no.setter
    def dc_tbl_no(self, dc_tbl_no: str):
        """dc_tbl_no setter."""
        self._dc_tbl_no = dc_tbl_no

    @property
    def ac_tbl_no(self) -> str:
        """ac_tbl_no getter."""
        return self._ac_tbl_no

    @ac_tbl_no.setter
    def ac_tbl_no(self, ac_tbl_no: str):
        """ac_tbl_no setter."""
        self._ac_tbl_no = ac_tbl_no

    @property
    def Ss(self) -> str:
        """Ss getter."""
        return self._Ss

    @Ss.setter
    def Ss(self, Ss: str):
        """Ss setter."""
        self._Ss = Ss

    @property
    def Se(self) -> str:
        """Se getter."""
        return self._Se

    @Se.setter
    def Se(self, Se: str):
        """Se setter."""
        self._Se = Se

    @property
    def Ah(self) -> str:
        """Ah getter."""
        return self._Ah

    @Ah.setter
    def Ah(self, Ah: str):
        """Ah setter."""
        self._Ah = Ah

    @property
    def Al(self) -> str:
        """Al getter."""
        return self._Al

    @Al.setter
    def Al(self, Al: str):
        """Al setter."""
        self._Al = Al

    def __repr__(self) -> str:
        return f'<Scan {self.Ss},{self.Se},{self.Ah},{self.Al}>'

    def __str__(self) -> str:
        """Converts the class to str, returns name."""
        return repr(self)

    # def __eq__(self, other) -> bool:
    #     """Compares two scans tables for equality."""
    #     return (self.bits == other.bits).all() and (self.values == other.values).all()
