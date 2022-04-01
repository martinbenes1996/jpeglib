
from dataclasses import dataclass
import numpy as np
import tempfile
from . import _jpeg

@dataclass
class DCTJPEG(_jpeg.JPEG):
    """JPEG instance to work in DCT domain."""
    Y: np.ndarray
    Cb: np.ndarray
    Cr: np.ndarray
    qt: np.ndarray
    
    def _read_dct(self):
        # write content into temporary file
        tmp = tempfile.NamedTemporaryFile(suffix='jpeg')
        tmp.write(self.content)
        # read DCT
        self._Y,self._Cb,self._Cr,self._qt = _jpeg.read_dct(tmp.name, self)
    
    @property
    def Y(self) -> np.ndarray:
        if self._Y is None:
            self._read_dct()
        return self._Y
    @Y.setter
    def Y(self, Y: np.ndarray):
        self._Y = Y
        
    @property
    def Cb(self) -> np.ndarray:
        if self._Cb is None:
            self._read_dct()
        return self._Cb
    @Cb.setter
    def Cb(self, Cb: np.ndarray):
        self._Cb = Cb
        
    @property
    def Cr(self, Cr: np.ndarray) -> np.ndarray:
        if self._Cr is None:
            self._read_dct()
        return self._Cr
    @Cr.setter
    def Cr(self, Cr: np.ndarray):
        self._Cr = Cr
        
    @property
    def qt(self, qt: np.ndarray) -> np.ndarray:
        if self._qt is None:
            self._read_dct()
        return self._qt
    @qt.setter
    def qt(self, qt: np.ndarray):
        self._qt = qt
    
    def _free(self):
        del self._Y
        del self._Cb
        del self._Cr
        del self._qt
    
