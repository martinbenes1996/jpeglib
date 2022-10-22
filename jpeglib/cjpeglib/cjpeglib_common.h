#ifndef CJPEGLIB_COMMON_HPP
#define CJPEGLIB_COMMON_HPP

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// this is envelope for jpeglib.h
// trying to avoid naming the same as system library
#include "vjpeglib.h"
#include "cjpeglib.h"
#include "cjpeglib_common_flags.h"
#include "cjpeglib_common_markers.h"

// declaration to avoid errors
long jround_up (long a, long b);

FILE *_read_jpeg(
	const char *filename,
    struct jpeg_decompress_struct *cinfo,
    struct jpeg_error_mgr *jerr,
    bool read_header
);

int read_jpeg_info(
  const char *srcfile,
  int *block_dims,
  int *image_dims,
  int *num_components,
  int *samp_factor,
  int *jpeg_color_space,
  int *marker_lengths,
  int *marker_types,
  BITMASK *flags
);


/**
 * @brief Custom error handler, mapping libjpeg error on C++ exception.
 * @author licy183
 *
 * @param cinfo
 */
void my_custom_error_exit(
	j_common_ptr cinfo
);
#define jpeg_std_error(jerr) ((jpeg_std_error(jerr)), ((jerr)->error_exit = my_custom_error_exit), (jerr))

void _write_qt(
	struct jpeg_compress_struct * cinfo,
	unsigned short *qt,
	short *quant_tbl_no,
	unsigned char only_create
);

int print_jpeg_params(
	const char *srcfile
);

#ifdef __cplusplus
}
#endif

#endif // CJPEGLIB_COMMON_HPP