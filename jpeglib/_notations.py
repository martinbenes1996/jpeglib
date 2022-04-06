
import math

def GCD(*args):
    """Computes greatest common divider (gcd) of the positional arguments."""
    # replace 0s with 1s
    args = [a if a != 0 else 1 for a in args]
    # all divisors
    divisors = [i for i in range(1,min(args)+1) if all([x % i == 0 for x in args])]
    # choose the maximal
    gcd = max(divisors)
    return gcd

def Jab_to_factors(Jab):
    """Converts J:a:b notation to the 6 factors, used by libjpeg.
    
    Read more about the notations in `Tuomas Siipola's article <https://zpl.fi/chroma-subsampling-and-jpeg-sampling-factors/>`_
    or `glossary <https://jpeglib.readthedocs.io/en/latest/glossary.html#jpeg-sampling-factor>`_.
    
    :param Jab: List of three numbers, J:a:b notation of subsampling.
    :type dstfile: list
    :return: Subsampling factors, as list of three lists, each containing 2 numbers.
    :rtype: list

    :Example:

    >>> Jab_to_factors([4,2,0]) # -> [[2,2],[1,1],[1,1]]
    """
    # parse input
    J,a,b = Jab
    assert(J in {1,2,3,4})
    if J == 4: assert(a in {4,2,1,0})
    elif J == 3: assert(a in {3,1,0})
    else: assert(a in {J,1,0})
    assert(b in {a,0})
    # chroma dimensions
    Y = [J,2]
    Cb = Cr = [a,(a == b) + 1]
    # normalize by GCD
    gcd0 = GCD(Y[0], Cb[0], Cr[0])
    gcd1 = GCD(Y[1], Cb[1], Cr[1])
    factors = [
        [int(Y[0] / gcd0), int(Y[1] / gcd1)],
        [int(Cb[0] / gcd0), int(Cb[1] / gcd1)],
        [int(Cr[0] / gcd0), int(Cr[1] / gcd1)],
    ]
    return factors
    
if __name__ == '__main__':
    Jab_to_factors([4,4,4])
    Jab_to_factors([4,4,0])
    Jab_to_factors([4,2,2])
    Jab_to_factors([4,2,0])
    Jab_to_factors([4,1,1])
    Jab_to_factors([4,1,0])
    Jab_to_factors([3,3,3])
    Jab_to_factors([3,3,0])
    Jab_to_factors([3,1,1])
    Jab_to_factors([3,1,0])
    