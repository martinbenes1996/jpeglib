[![PyPI version](https://badge.fury.io/py/jpeglib.svg)](https://pypi.org/project/jpeglib/)
[![Commit CI/CD](https://github.com/martinbenes1996/jpeglib/actions/workflows/on_commit.yml/badge.svg?branch=master)](https://github.com/martinbenes1996/jpeglib/actions/workflows/on_commit.yml)
[![Release CI/CD](https://github.com/martinbenes1996/jpeglib/actions/workflows/on_release.yml/badge.svg)](https://github.com/martinbenes1996/jpeglib/actions/workflows/on_release.yml)
[![Documentation Status](https://readthedocs.org/projects/jpeglib/badge/?version=latest)](https://jpeglib.readthedocs.io/)
[![PyPI downloads](https://img.shields.io/pypi/dm/jpeglib)](https://pypi.org/project/jpeglib/)
[![Stars](https://img.shields.io/github/stars/martinbenes1996/jpeglib.svg)](https://GitHub.com/martinbenes1996/jpeglib)
[![Contributors](https://img.shields.io/github/contributors/martinbenes1996/jpeglib)](https://GitHub.com/martinbenes1996/jpeglib)
[![Wheel](https://img.shields.io/pypi/wheel/jpeglib)](https://pypi.org/project/jpeglib/)
[![Status](https://img.shields.io/pypi/status/jpeglib)](https://pypi.com/project/jpeglib/)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/jpeglib/)
[![Last commit](https://img.shields.io/github/last-commit/martinbenes1996/jpeglib)](https://GitHub.com/martinbenes1996/jpeglib)


# jpeglib

Python envelope for the popular C library libjpeg for handling JPEG files.

*libjpeg* offers full control over compression and decompression and exposes DCT coefficients and quantization tables.

## Installation

Simply install the package with pip3


```bash
pip install jpeglib
```


> :warning: This will install `jpeglib` together with every integrated version of libjpeg, libjpeg-turbo and mozjpeg. It takes longer to install than the package.


## Usage

Import the library in Python 3

```python
import jpeglib
```

### DCT

Get *discrete cosine transform* (DCT) coefficients and quantization matrices as numpy array


```python
im = jpeglib.read_dct("input.jpeg")
im.Y; im.Cb; im.Cr; im.qt
```

You get luminance DCT, chrominance DCT and quantization tables.

Write the DCT coefficients back to a file with

```python
im.write_dct("output.jpeg")
```

### Pixel data

Decompress the `input.jpeg` into spatial representation in numpy array with

```python
im = jpeglib.read_spatial("input.jpeg")
im.spatial
```

You can specify parameters such as output color space, DCT method, dithering, etc.

Write spatial representation in numpy arrray back to file with

```python
im.write_spatial("output.jpeg")
```

You can specify input color space, DCT method, sampling factor, output quality, smoothing factor etc.

You can find all the details in the [documentation](https://jpeglib.readthedocs.io/).

### libjpeg version

It is possible to choose, which version of libjpeg should be used.

```python
jpeglib.version.set('6b')
```

Currently `jpeglib` supports all versions of libjpeg from 6b to 9e, libjpeg-turbo 2.1.0 and mozjpeg 4.0.3.
Their source codes is baked inside the package and thus distributed with it, avoiding external dependency.

Get currently used libjpeg version by

```python
version = jpeglib.version.get()
```

You can also set a libjpeg version for a scope only.

```python
jpeglib.version.set('6b')
im = jpeglib.read_spatial('image.jpeg') # using 6b
with jpeglib.version('9e'):
    im = jpeglib.read_spatial('image.jpeg') # using 9e
im = jpeglib.read_spatial('image.jpeg') # using 6b again
```


## Credits

Developed by [Martin Benes](https://github.com/martinbenes1996).
