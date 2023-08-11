
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// this is envelope for jpeglib.h
// trying to avoid naming the same as system library
#include "vjpeglib.h"
#include "cjpeglib.h"
#include "cjpeglib_common.h"


int read_jpeg_spatial(
	const char *srcfile,
	unsigned char *rgb,
	unsigned char *colormap, // colormap used
	unsigned char *in_colormap, // colormap to use
	int out_color_space,
	int dither_mode,
	int dct_method,
	BITMASK flags
) {
	// allocate
	FILE *fp;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// read jpeg header
		if((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL) {
			return 0;
		}

		// set parameters
		if(out_color_space >= 0) {
			cinfo.out_color_space = (J_COLOR_SPACE)out_color_space;
		} else {
			cinfo.out_color_space = (J_COLOR_SPACE)cinfo.jpeg_color_space;
		}
		if(dither_mode >= 0) {
			cinfo.dither_mode = (J_DITHER_MODE)dither_mode;
		}
		if(dct_method >= 0) {
			cinfo.dct_method = (J_DCT_METHOD)dct_method;
		}
		if (overwrite_flag(flags, DO_FANCY_UPSAMPLING)) {
			cinfo.do_fancy_upsampling = flag_is_set(flags, DO_FANCY_UPSAMPLING);
		}
		if (overwrite_flag(flags, DO_BLOCK_SMOOTHING)) {
			cinfo.do_block_smoothing = flag_is_set(flags, DO_BLOCK_SMOOTHING);
		}
		if (overwrite_flag(flags, QUANTIZE_COLORS)) {
			cinfo.quantize_colors = flag_is_set(flags, QUANTIZE_COLORS);
		}
		unsigned char *cmap[256];
		if (in_colormap != NULL)
			for (int i = 0; i < 256; i++) {
				cmap[i] = in_colormap + i * 3;
			}

		if (overwrite_flag(flags, QUANTIZE_COLORS) && flag_is_set(flags, QUANTIZE_COLORS)) {
			cinfo.actual_number_of_colors = 256; // TODO: parametrized
			cinfo.desired_number_of_colors = 256;
			if (in_colormap != NULL)
				cinfo.colormap = (JSAMPARRAY)cmap;
		}

		if (overwrite_flag(flags, PROGRESSIVE_MODE)) {
			cinfo.progressive_mode = flag_is_set(flags, PROGRESSIVE_MODE);
		}
		if (overwrite_flag(flags, ARITH_CODE)) {
			cinfo.arith_code = flag_is_set(flags, ARITH_CODE);
		}
		if (overwrite_flag(flags, CCIR601_SAMPLING)) {
			cinfo.CCIR601_sampling = flag_is_set(flags, CCIR601_SAMPLING);
		}
		if (overwrite_flag(flags, TWO_PASS_QUANTIZE)) {
			cinfo.two_pass_quantize = flag_is_set(flags, TWO_PASS_QUANTIZE);
		}
		if (overwrite_flag(flags, ENABLE_1PASS_QUANT)) {
			cinfo.enable_1pass_quant = flag_is_set(flags, ENABLE_1PASS_QUANT);
		}
		if (overwrite_flag(flags, ENABLE_EXTERNAL_QUANT)) {
			cinfo.enable_external_quant = flag_is_set(flags, ENABLE_EXTERNAL_QUANT);
		}
		if (overwrite_flag(flags, ENABLE_2PASS_QUANT)) {
			cinfo.enable_2pass_quant = flag_is_set(flags, ENABLE_2PASS_QUANT);
		}
		// decompress
		(void)jpeg_start_decompress(&cinfo);
		// read pixels
		char *rowptr = (char *)rgb;
		unsigned short stride = cinfo.out_color_components;
		if(overwrite_flag(flags, QUANTIZE_COLORS) && flag_is_set(flags, QUANTIZE_COLORS))
			stride = 1;

		while (cinfo.output_scanline < cinfo.output_height) {
			jpeg_read_scanlines(
				&cinfo,
				(JSAMPARRAY)&rowptr,
				1
			);
			rowptr += cinfo.output_width * stride;
		}

		// cleanup
		(void)jpeg_finish_decompress(&cinfo);

	// error handling
	} catch(...) {
		status = 0;
	}

	// cleanup and return
	jpeg_destroy_decompress(&cinfo);
	if(fp != NULL)
		fclose(fp);
	return status;
}

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
) {

	// allocate
	FILE *fp;
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	memset(&cinfo, 0x00, sizeof(struct jpeg_compress_struct));
	memset(&jerr, 0x00, sizeof(struct jpeg_error_mgr));

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// open the destination file
		if ((fp = fopen(dstfile, "wb")) == NULL) {
			fprintf(stderr, "can't open %s\n", dstfile);
			return 0;
		}
		cinfo.err = jpeg_std_error(&jerr);
		jpeg_create_compress(&cinfo);
		jpeg_stdio_dest(&cinfo, fp);

		// set basic parameters
		cinfo.image_height = image_dims[0];
		cinfo.image_width = image_dims[1];
		if (jpeg_color_space != NULL) {
			cinfo.in_color_space = (J_COLOR_SPACE)(jpeg_color_space[0]);
		}
		if (num_components != NULL) {
			cinfo.input_components = num_components[0];
		}
		jpeg_set_defaults(&cinfo);
		jpeg_set_colorspace(&cinfo, (J_COLOR_SPACE)(jpeg_color_space[1]));

		// set advanced parameters
		if (dct_method >= 0) {
			cinfo.dct_method = (J_DCT_METHOD)dct_method;
		}
		int chroma_factor[2];
		if (samp_factor != NULL) {
			chroma_factor[0] = *(samp_factor + 0);
			chroma_factor[1] = *(samp_factor + 1);
			for (int ch = 0; ch < cinfo.num_components; ch++) {
				cinfo.comp_info[ch].v_samp_factor = *(samp_factor + ch * 2 + 0);
				cinfo.comp_info[ch].h_samp_factor = *(samp_factor + ch * 2 + 1);
			}
		} else {
			chroma_factor[0] = cinfo.comp_info[0].h_samp_factor;
			chroma_factor[1] = cinfo.comp_info[0].v_samp_factor;
		}

		// for(int i = 0; i < 4; i++) {
		// 	fprintf(stderr, "%d) %p %p\n",
		// 		i,
		// 		cinfo.dc_huff_tbl_ptrs[i],
		// 		cinfo.ac_huff_tbl_ptrs[i]
		// 	);
		// }

		// write qt
		if(qt != NULL) {
			_write_qt(&cinfo, qt, quant_tbl_no, 1);
		// write quality
		} else if (quality > 0) {
			#if LIBVERSION >= 6300
			if(base_quant_tbl_idx >= 0) {
				jpeg_c_set_int_param(
					&cinfo,
					JINT_BASE_QUANT_TBL_IDX,
					base_quant_tbl_idx
				);
			}
			#endif

			// force baseline (8bit quantization)
			boolean force_baseline = FALSE;
			if (overwrite_flag(flags, FORCE_BASELINE))
				force_baseline = flag_is_set(flags, FORCE_BASELINE);
			jpeg_set_quality(&cinfo, quality, force_baseline);
		}

		if (smoothing_factor >= 0) {
			cinfo.smoothing_factor = smoothing_factor;
		}
		// if (in_color_space >= 0)
		//   cinfo.in_color_space = in_color_space;

		#if LIBVERSION >= 70
		if (overwrite_flag(flags, DO_FANCY_UPSAMPLING)) {
			cinfo.do_fancy_downsampling = flag_is_set(flags, DO_FANCY_UPSAMPLING);
		}
		#endif
		if (overwrite_flag(flags, PROGRESSIVE_MODE)) {
			cinfo.progressive_mode = flag_is_set(flags, PROGRESSIVE_MODE);
		}
		if (overwrite_flag(flags, PROGRESSIVE_MODE) && flag_is_set(flags, PROGRESSIVE_MODE)) {
			jpeg_simple_progression(&cinfo);
		}
		if (overwrite_flag(flags, OPTIMIZE_CODING)) {
			cinfo.optimize_coding = flag_is_set(flags, OPTIMIZE_CODING);
		}
		#ifdef C_ARITH_CODING_SUPPORTED
		if (overwrite_flag(flags, ARITH_CODE)) {
			cinfo.arith_code = flag_is_set(flags, ARITH_CODE);
		}
		#endif
		if (overwrite_flag(flags, WRITE_JFIF_HEADER)) {
			cinfo.write_JFIF_header = flag_is_set(flags, WRITE_JFIF_HEADER);
		}
		if (overwrite_flag(flags, WRITE_ADOBE_MARKER)) {
			cinfo.write_Adobe_marker = flag_is_set(flags, WRITE_ADOBE_MARKER);
		}
		if (overwrite_flag(flags, CCIR601_SAMPLING)) {
			cinfo.CCIR601_sampling = flag_is_set(flags, CCIR601_SAMPLING);
		}
		#if LIBVERSION >= 6300
		if(overwrite_flag(flags, TRELLIS_QUANT)) {
			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_TRELLIS_QUANT,
				flag_is_set(flags, TRELLIS_QUANT)
			);
		}
		if(overwrite_flag(flags, TRELLIS_QUANT_DC)) {
			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_TRELLIS_QUANT_DC,
				flag_is_set(flags, TRELLIS_QUANT_DC)
			);
		}
		#endif

		jpeg_start_compress(&cinfo, TRUE);

		// write markers
		int offset = 0;
		for (int i = 0; i < num_markers; i++) {
			jpeg_write_marker(
				&cinfo,
				marker_types[i],
				(const JOCTET *)(markers + offset),
				// markers + offset,
				marker_lengths[i]
			);
			offset += marker_lengths[i];
		}

		// write data
		unsigned char *rowptr = rgb;
		for (unsigned h = 0; h < cinfo.image_height; h++) {
			jpeg_write_scanlines(&cinfo, &rowptr, 1);
			rowptr += cinfo.image_width * cinfo.input_components;
		}
		// cleanup
		jpeg_finish_compress(&cinfo);

	// error handling
	} catch(...) {
		status = 0;
	}

	// cleanup and return
	jpeg_destroy_compress(&cinfo);
	if(fp != NULL)
		fclose(fp);
	return status;
}


#ifdef __cplusplus
}
#endif