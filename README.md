# jpeglib

[![PyPI version](https://badge.fury.io/py/jpeglib.svg)](https://badge.fury.io/py/jpeglib)
[![Documentation Status](https://readthedocs.org/projects/jpeglib/badge/?version=latest)](https://jpeglib.readthedocs.io/en/latest/?badge=latest)

Python envelope for the popular C library libjpeg for handling JPEG files.

Currently almost all of the popular Python image libraries use *libjpeg* under the hood, however do not expose the whole spectrum of parameters that libjpeg offers. At the same time it is usually also impossible to load a low-level JPEG components - DCT coefficients and quantization tables. All of this is possible with `jpeglib`.

## Setup

Package `jpeglib` is developed for Python3. Install using pip (or pip3) with

```bash
pip3 install jpeglib
```

## Usage

There is a file `input.jpeg` of dimensions H x W. Get its DCT (*discrete cosine transform*) coefficients and quantization matrices in numpy tensor with

```python
import stegojpeg
im = stegojpeg.JPEG("input.jpeg") # load metadata
Y,CbCr,qt = im.read_dct() # load data
```

We receive three values

* **Luminance DCT coefficients** as numpy array of shape (1,H/8,W/8,8,8)
* **Chrominance DCT coefficients** as numpy array of shape (2,H/8/2,W/8/2,8,8) for subsampling 4:2:2
* **Quantization tables** as numpy array of shape (3,8,8)

DCT coefficients are commonly used to embed a steganographic message to. Once done, create a file with modified DCT with

```python
im.write_dct("output.jpeg", Y, CbCr) # write data
```

Read the spatial representation of `input.jpeg` with

```python
im = stegojpeg.JPEG("input.jpeg")
spatial = im.read_spatial()
```

The reading can be parameterized with spatial color space, dithering mode, dct method and various boolean flags.
For not-specified parameters, default of libjpeg is used. 

```python
spatial = im.read_spatial(
  out_color_space = "JCS_GRAYSCALE", # color space of output
  dither_mode     = "JDITHER_FS",    # dithering mode
  dct_method      = "JDCT_FLOAT",    # dct method
  flags           = ["DO_FANCY_UPSAMPLING", "DO_BLOCK_SMOOTHING"] # flags to be set true
)
```

Writing of a spatial representation is done with

```python
im.write_spatial("output.jpeg", spatial)
```

If you do not specify the second parameter, the original source data is used.

Similarly to reading there are parameters for writing that you can use.
Based on the source of libjpeg default, if not specified.

```python
im.write_spatial(
  "output.jpeg",
  spatial,
  in_color_space   = "JCS_RGB",    # color space of input
  dct_method       = "JDCT_ISLOW", # dct method
  samp_factor      = (2,1,1),      # sampling factor
  quality          = 75,           # compression quality
  smoothing_factor = 0,            # smoothing factor
  flags            = ["QUANTIZE_COLORS","WRITE_JFIF_HEADER"]
)
```

It is possible to choose, which version of libjpeg should be used. For now only `6b` and `8d` are supported.

```python

```

## Dependencies and platforms

Tested on MacOS and Linux. There is a [known bug](https://stackoverflow.com/questions/49299905/error-lnk2001-unresolved-external-symbol-pyinit) which is planned to be fixed in the future.

For the installation you need

* C compiler (gcc, clang)
* [libjpeg](http://libjpeg.sourceforge.net/) + properly configured environment (so compiler & linker can find it)

## How is the libjpeg loaded

The *libjpeg* source code is baked inside the `jpeglib` package, thus there are only Python dependencies, such as numpy.

## Future extensions

* **Possible to make changes in quantization tables** = for now modifications are only possible on DCT coefficients

* **Independence on source file** = to write matrices you need the object you used for reading, because metadata are copied. I would like this to be independent, so the writing is state-less and the memory can be released

```python
import stegojpeg
with stegojpeg.JPEG("input.jpeg") as im:
    Y,CbCr,qt = im.read_dct() # read DCT

    # modify here
    
    im.write_dct("output.jpeg", Y, CbCr) # write DCT

# load RGB
import numpy as np
from PIL import Image
im1 = np.array(Image.open("input.jpeg")) # original
im2 = np.array(Image.open("output.jpeg")) # modified
```

* **Fix Windows installation** = make sure the library can be installed on Windows

## Credits

Developed by [Martin Benes](https://github.com/martinbenes1996).