
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib.h"
#include "cjpeglib_common_markers.h"

/**
 * Because processing of markers is asynchronous,
 * information about markers have to be temporarily stored in global variables.
 *
 */
int gpos = 0;
int gmarker_types[MAX_MARKER];
int gmarker_lengths[MAX_MARKER];
unsigned char * gmarker_data[MAX_MARKER];

FILE *_read_jpeg(
	const char *filename,
    struct jpeg_decompress_struct *cinfo,
    struct jpeg_error_mgr *jerr,
    bool read_header
);

int read_jpeg_markers(
  const char *srcfile,
  unsigned char *markers
) {
	// allocate
	FILE *fp = NULL;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// read jpeg header
		if ((fp = _read_jpeg(srcfile, &cinfo, &jerr, FALSE)) == NULL)
			return 0;

		// markers
		if (markers != NULL) {
			// set marker handlers
			set_marker_handlers(&cinfo);
			// read markers
			(void)jpeg_read_header(&cinfo, TRUE);
			// collect marker data
			int offset = 0;
			for (int i = 0; i < gpos; i++) {
				for (int j = 0; j < gmarker_lengths[i]; j++) {
					markers[offset + j] = gmarker_data[i][j];
				}
				offset += gmarker_lengths[i];
			}
			unset_marker_handlers(&cinfo);
		}

	// error handling
	} catch(...) {
		status = 0;
	}

	// cleanup
	jpeg_destroy_decompress(&cinfo);
	if(fp != NULL)
		fclose(fp);
	return status;
}


int jpeg_getc(
	j_decompress_ptr cinfo
) {
	struct jpeg_source_mgr *datasrc = cinfo->src;
	// no bytes in buffer
	if (datasrc->bytes_in_buffer == 0) {
		if (!(*datasrc->fill_input_buffer)(cinfo))
		return -1; // return error
	}
	// read char
	datasrc->bytes_in_buffer--;
	return GETJOCTET(*datasrc->next_input_byte++);
}

boolean jpeg_handle_marker(
	j_decompress_ptr cinfo
) {
	// get marker name
	char mname[20];
	if (cinfo->unread_marker == JPEG_COM)
		sprintf(mname, "COM");
	else sprintf(mname, "APP%d", cinfo->unread_marker - JPEG_APP0);

	// get length
	unsigned char *p = NULL;
	int length = 0;
	length = jpeg_getc(cinfo) << 8;
	length += jpeg_getc(cinfo);
	length -= 2; // discount the length word itself
	gmarker_lengths[gpos] = length;

	// allocate
	if (gpos < MAX_MARKER) {
		gmarker_types[gpos] = cinfo->unread_marker;
		// strcpy(mark_name[gpos], mname);
		p = new unsigned char[length + 1];
		// if ((p = malloc((length + 1) * sizeof(char))) == NULL) {
		// 	fprintf(stderr, "Bad malloc!\n");
		// 	return FALSE;
		// }
		gmarker_data[gpos] = p;
		gpos += 1;

	// too many markers
	} else {
		fprintf(stderr, "Too many markers - %s [%d] skipped\n", mname, gpos);
		return FALSE;
	}

	// iterate over data
	int c;
	while (--length >= 0) {
		if ((c = jpeg_getc(cinfo)) == -1) {
			fprintf(stderr, "Error parsing marker %s\n", mname);
			return FALSE;
		}
		*(p) = (unsigned char)c;
		// if (!p[0]) p[0] = 0x20; // replace 0x0 byte with 0x20
		p++;
	}
	p[0] = 0; // set the last byte to 0

	return TRUE;
}

int set_marker_handlers(
	struct jpeg_decompress_struct *cinfo
) {
	// jpeg globals
	gpos = 0;
	for (int i = 0; i < MAX_MARKER; i++) {
		gmarker_types[i] = 0;
		gmarker_data[i] = NULL;
		gmarker_lengths[i] = 0;
	}
	// set handlers
	jpeg_set_marker_processor(
		cinfo,
		JPEG_COM,
		jpeg_handle_marker
	);
	// jpeg_set_marker_processor(
	// 	cinfo,
	// 	JPEG_APP0 + 15,
	// 	jpeg_handle_marker
	// );
	for (int i = 0; i <= 15; i++)
		jpeg_set_marker_processor(
			cinfo,
			JPEG_APP0 + i,
			jpeg_handle_marker
		);

	return 1;
}

int unset_marker_handlers(
	struct jpeg_decompress_struct *cinfo
) {
	// jpeg globals
	for (int i = 0; i < gpos; i++) {
		gmarker_lengths[i] = 0;
		gmarker_types[i] = 0;
		if (gmarker_data[i] != NULL)
			delete [] gmarker_data[i];
			// free((void *)gmarker_data[i]);
	}
	gpos = 0;

	// set handlers
	(void)cinfo;

	return 1;
}

#ifdef __cplusplus
}
#endif