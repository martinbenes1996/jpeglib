#ifndef CJPEGLIB_COMMON_MARKERS_HPP
#define CJPEGLIB_COMMON_MARKERS_HPP

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// this is envelope for jpeglib.h
// trying to avoid naming the same as system library
#include "vjpeglib.h"

/**
 * Because processing of markers is asynchronous,
 * information about markers have to be temporarily stored in global variables.
 *
 */
#define MAX_MARKER 50  // limit for markers
static int gpos = 0;
static int gmarker_types[MAX_MARKER];
static int gmarker_lengths[MAX_MARKER];
static unsigned char * gmarker_data[MAX_MARKER];

/**
 * @brief Set the marker handlers for reading in decompression.
 *
 * @param cinfo
 * @return int
 */
int set_marker_handlers(
	struct jpeg_decompress_struct * cinfo
);
/**
 * @brief Deallocated the marker handlers.
 *
 * @param cinfo
 * @return int
 */
int unset_marker_handlers(
	struct jpeg_decompress_struct * cinfo
);
// /**
//  * @brief Reads the next byte of marker.
//  *
//  * @param cinfo
//  * @return int
//  */
// int jpeg_getc (
// 	j_decompress_ptr cinfo
// );
// /**
//  * @brief Asynchronous marker handler.
//  *
//  * @param cinfo
//  * @return int
//  */
// int jpeg_handle_marker (
// 	j_decompress_ptr cinfo
// );

int read_jpeg_markers(
  const char *srcfile,
  unsigned char *markers
);

#ifdef __cplusplus
}
#endif

#endif // CJPEGLIB_COMMON_MARKERS_HPP