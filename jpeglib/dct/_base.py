
import numpy as np
from typing import Union

def Lambda(x:Union[float,int,list,np.ndarray]) -> np.array:
    """DCT scaling function.
    
    :param x: Input value or array.
    :type x: float | list | np.ndarray
    :return: sqrt(1/2) for x=0, otherwise 1
    :rtype: float
    
    :Examples:
    
    Example of call with scalar
    
    >>> Lambda(0) # -> [0.70710678]
    
    Example of call with vector
    
    >>> Lambda([-1,0,1]) # -> [1. 0.70710678 1.]
    """
    y = np.array(x, dtype=np.float64)
    y[y != 0] = 1.
    y[y == 0] = 1/np.sqrt(2)
    return y

