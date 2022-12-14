#ifndef CJPEGLIB_H
#define CJPEGLIB_H

#ifdef _WIN32
#define LIBRARY_API extern "C" __declspec(dllexport)
#else
#define LIBRARY_API extern "C"
#endif


#include "cjpeglib_common_flags.h"

// ---------- Meta -------------
LIBRARY_API
int read_jpeg_info(
    const char *srcfile,
    int *block_dims,
    int *image_dims,
    int *num_components,
    int *samp_factor,
    int *jpeg_color_space,
    int *marker_lengths,
    int *mark_types,
	unsigned char *huffman_valid,
    unsigned char *huffman_bits,
    unsigned char *huffman_values,
    BITMASK *flags
);

LIBRARY_API
int read_jpeg_markers(
    const char *srcfile,
    unsigned char *markers
);

// ----------- DCT -------------
LIBRARY_API
int read_jpeg_dct(
    const char *srcfile,
    short *Y,
    short *Cb,
    short *Cr,
    short *K,
    unsigned short *qt,
    unsigned char *quant_tbl_no
);
LIBRARY_API
int write_jpeg_dct(
    const char *srcfile,
    const char *dstfile,
    short *Y,
    short *Cb,
    short *Cr,
    short *K,
    int *image_dims,
    int *block_dims,
    int in_color_space,
    int in_components,
    unsigned short *qt,
    short quality,
    short *quant_tbl_no,
    int num_markers,
    int *marker_types,
    int *marker_lengths,
    unsigned char *markers,
	BITMASK flags
);

// ----------- RGB -------------
LIBRARY_API
int read_jpeg_spatial(
    const char *srcfile,
    unsigned char *rgb,
    unsigned char *colormap,    // colormap used
    unsigned char *in_colormap, // colormap to use
    int out_color_space,
    int dither_mode,
    int dct_method,
    BITMASK flags
);

LIBRARY_API
int write_jpeg_spatial(
    const char *dstfile,
    unsigned char *rgb,
    int *image_dims,
    int *jpeg_color_space,
    int *num_components,
    int dct_method,
    int *samp_factor,
    unsigned short *qt,
    short quality,
    short *quant_tbl_no,
    short base_quant_tbl_idx,
    short smoothing_factor,
    int num_markers,
    int *marker_types,
    int *marker_lengths,
    unsigned char *markers,
    BITMASK flags
);

// int jpeg_lib_version(void) { return JPEG_LIB_VERSION; }
LIBRARY_API
int print_jpeg_params(const char *srcfile);


#endif // CJPEGLIB_H