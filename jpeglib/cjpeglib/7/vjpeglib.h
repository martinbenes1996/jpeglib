#ifndef VJPEGLIB_H
#define VJPEGLIB_H

// public API
#ifndef JPEG_INTERNALS
#include "jpeglib.h"

// private API
#else
#include "jcdctmgr.c"
#endif

#endif // _VJPEGLIB_H_