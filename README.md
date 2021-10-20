[![PyPI version](https://badge.fury.io/py/jpeglib.svg)](https://pypi.org/project/jpeglib/)
[![Documentation Status](https://readthedocs.org/projects/jpeglib/badge/?version=latest)](https://jpeglib.readthedocs.io/)
[![GitHub](https://img.shields.io/github/stars/martinbenes1996/jpeglib.svg)](https://GitHub.com/martinbenes1996/jpeglib)
[![PyPi license](https://badgen.net/pypi/license/pip/)](https://pypi.com/project/jpeglib/)
![Unittests](https://github.com/martinbenes1996/jpeglib/actions/workflows/unittests_on_commit.yml/badge.svg)

# jpeglib

Python envelope for the popular C library libjpeg for handling JPEG files.

*libjpeg* offers full control over compression and decompression and exposes DCT coefficients and quantization tables.

## Installation

Simply install the package with pip3


```bash
pip install jpeglib
```

## Usage

Import the library in Python 3

```python
import jpeglib
```

### DCT

Get *discrete cosine transform* (DCT) coefficients and quantization matrices as numpy array


```python
im = jpeglib.JPEG("input.jpeg") # load metadata
Y,CbCr,qt = im.read_dct() # load data
```

You get luminance DCT, chrominance DCT and quantization tables.

Write the DCT coefficients back to a file with

```python
im.write_dct("output.jpeg", Y, CbCr) # write data
```

You can also write the read-write sequence using `with` statement

```python
with jpeglib.JPEG("input.jpeg") as im:
  Y,CbCr,qt = im.read_dct()
  # modify the DCT coefficients
  im.write_dct("output.jpeg", Y, CbCr)
```

### Pixel data

Decompress the `input.jpeg` into spatial representation in numpy array with

```python
im = jpeglib.JPEG("input.jpeg")
rgb = im.read_spatial()
```

You can specify parameters such as output color space, DCT method, dithering, etc.

Write spatial representation in numpy arrray back to file with

```python
im.write_spatial("output.jpeg", spatial)
```

Here you can specify input color space, DCT method, sampling factor, output quality, smoothing factor etc.

You can find all the details in the [documentation](https://jpeglib.readthedocs.io/).

### libjpeg version

It is possible to choose, which version of libjpeg should be used.

```python
jpeglib.version.set('6b')
```

Currently `jpeglib` supports the most popular versions 6b and 8d. Their source codes is baked inside the package
and thus distributed with it, avoiding external dependency.

Get currently used libjpeg version by

```python
version = jpeglib.version.get()
```


## Credits

Developed by [Martin Benes](https://github.com/martinbenes1996).