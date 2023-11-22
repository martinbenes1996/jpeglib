"""

Author: Martin Benes
Affiliation: University of Innsbruck
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
    # Description: https://gitlab.linphone.org/BC/public/external/libjpeg-turbo/-/blob/bc/wizard.txt

    # def __post_init__(self):
    #     print(self.dc_tbl_no)

    @property
    def components(self) -> np.ndarray:
        """Components getter."""
        return self._components

    @components.setter
    def components(self, components: np.ndarray):
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
        scan_params = f"{self.Ss},{self.Se},{self.Ah},{self.Al}"
        return f"<Scan {scan_params}: {self.components}>"

    def __str__(self) -> str:
        """Converts the class to str, returns name."""
        return repr(self)

    def __eq__(self, other) -> bool:
        """Compares two scans for equality."""
        return all(
            [
                (self._components == other.components).all(),
                self._Ss == other.Ss,
                self._Se == other.Se,
                self._Ah == other.Ah,
                self._Al == other.Al,
            ]
        )
