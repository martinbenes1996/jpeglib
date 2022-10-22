#ifndef CJPEGLIB_COMMON_MARKERS_HPP
#define CJPEGLIB_COMMON_MARKERS_HPP

#ifdef __cplusplus
extern "C" {
#endif

// this is envelope for jpeglib.h
// trying to avoid naming the same as system library
#include "vjpeglib.h"

#define MAX_MARKER 50  // limit for markers

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
/**
 * @brief Reads the next byte of marker.
 *
 * @param cinfo
 * @return int
 */
int jpeg_getc (
	j_decompress_ptr cinfo
);
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