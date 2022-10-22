extern "C" {

#include <stdio.h>
#include "vjpeglib.h"
#include "cjpeglib.h"

void my_custom_error_exit(j_common_ptr cinfo) {
  /* Write the message */
  (*cinfo->err->output_message)(cinfo);

  /* Let the memory manager delete any temp files before we die */
  jpeg_destroy(cinfo);

  /* Throw an "exception" using C++ */
  throw 0;
}

extern int read_jpeg_info_impl(const char *srcfile, int *block_dims,
                               int *image_dims, int *num_components,
                               int *samp_factor, int *jpeg_color_space,
                               int *marker_lengths, int *mark_types,
                               BITMASK *flags);

int read_jpeg_info(const char *srcfile, int *block_dims, int *image_dims,
                   int *num_components, int *samp_factor, int *jpeg_color_space,
                   int *marker_lengths, int *mark_types, BITMASK *flags) {
  try {
    return read_jpeg_info_impl(srcfile, block_dims, image_dims, num_components,
                               samp_factor, jpeg_color_space, marker_lengths,
                               mark_types, flags);
  } catch (...) {
    return 0;
  }
}

extern int read_jpeg_markers_impl(const char *srcfile, unsigned char *markers);

int read_jpeg_markers(const char *srcfile, unsigned char *markers) {
  try {
    return read_jpeg_markers_impl(srcfile, markers);
  } catch (...) {
    return 0;
  }
}

extern int read_jpeg_dct_impl(const char *srcfile, short *Y, short *Cb,
                              short *Cr, unsigned short *qt,
                              unsigned char *quant_tbl_no);

int read_jpeg_dct(const char *srcfile, short *Y, short *Cb, short *Cr,
                  unsigned short *qt, unsigned char *quant_tbl_no) {
  try {
    return read_jpeg_dct_impl(srcfile, Y, Cb, Cr, qt, quant_tbl_no);
  } catch (...) {
    return 0;
  }
}

extern int write_jpeg_dct_impl(const char *srcfile, const char *dstfile,
                               short *Y, short *Cb, short *Cr, int *image_dims,
                               int *block_dims, int in_color_space,
                               int in_components, unsigned short *qt,
                               short quality, short *quant_tbl_no,
                               int num_markers, int *marker_types,
                               int *marker_lengths, unsigned char *markers);

int write_jpeg_dct(const char *srcfile, const char *dstfile, short *Y,
                   short *Cb, short *Cr, int *image_dims, int *block_dims,
                   int in_color_space, int in_components, unsigned short *qt,
                   short quality, short *quant_tbl_no, int num_markers,
                   int *marker_types, int *marker_lengths,
                   unsigned char *markers) {
  try {
    return write_jpeg_dct(srcfile, dstfile, Y, Cb, Cr, image_dims, block_dims,
                          in_color_space, in_components, qt, quality,
                          quant_tbl_no, num_markers, marker_types,
                          marker_lengths, markers);
  } catch (...) {
    return 0;
  }
}

extern int read_jpeg_spatial_impl(const char *srcfile, unsigned char *rgb,
                                  unsigned char *colormap,
                                  unsigned char *in_colormap,
                                  int out_color_space, int dither_mode,
                                  int dct_method, BITMASK flags);

int read_jpeg_spatial(const char *srcfile, unsigned char *rgb,
                      unsigned char *colormap, unsigned char *in_colormap,
                      int out_color_space, int dither_mode, int dct_method,
                      BITMASK flags) {
  try {
    return read_jpeg_spatial_impl(srcfile, rgb, colormap, in_colormap,
                                  out_color_space, dither_mode, dct_method,
                                  flags);
  } catch (...) {
    return 0;
  }
}

extern int write_jpeg_spatial_impl(const char *dstfile, unsigned char *rgb,
                                   int *image_dims, int *jpeg_color_space,
                                   int *num_components, int dct_method,
                                   int *samp_factor, unsigned short *qt,
                                   short quality, short *quant_tbl_no,
                                   short smoothing_factor, int num_markers,
                                   int *marker_types, int *marker_lengths,
                                   unsigned char *markers, BITMASK flags);

int write_jpeg_spatial(const char *dstfile, unsigned char *rgb, int *image_dims,
                       int *jpeg_color_space, int *num_components,
                       int dct_method, int *samp_factor, unsigned short *qt,
                       short quality, short *quant_tbl_no,
                       short smoothing_factor, int num_markers,
                       int *marker_types, int *marker_lengths,
                       unsigned char *markers, BITMASK flags) {
  try {
    return write_jpeg_spatial(
        dstfile, rgb, image_dims, jpeg_color_space, num_components, dct_method,
        samp_factor, qt, quality, quant_tbl_no, smoothing_factor, num_markers,
        marker_types, marker_lengths, markers, flags);
  } catch (...) {
    return 0;
  }
}

extern int print_jpeg_params_impl(const char *srcfile);

int print_jpeg_params(const char *srcfile) {
  try {
    return print_jpeg_params(srcfile);
  } catch (...) {
    return 0;
  }
}

}