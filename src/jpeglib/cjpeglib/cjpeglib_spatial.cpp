
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
#include "jpegint.h"


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
		if (overwrite_default(flags, DO_FANCY_UPSAMPLING)) {
			cinfo.do_fancy_upsampling = flag_to_boolean_value(flags, DO_FANCY_UPSAMPLING);
		}
		if (overwrite_default(flags, DO_BLOCK_SMOOTHING)) {
			cinfo.do_block_smoothing = flag_to_boolean_value(flags, DO_BLOCK_SMOOTHING);
		}
		if (overwrite_default(flags, QUANTIZE_COLORS)) {
			cinfo.quantize_colors = flag_to_boolean_value(flags, QUANTIZE_COLORS);
		}
		unsigned char *cmap[256];
		if (in_colormap != NULL)
			for (int i = 0; i < 256; i++) {
				cmap[i] = in_colormap + i * 3;
			}

		if (overwrite_default(flags, QUANTIZE_COLORS) && flag_to_boolean_value(flags, QUANTIZE_COLORS)) {
			cinfo.actual_number_of_colors = 256; // TODO: parametrized
			cinfo.desired_number_of_colors = 256;
			if (in_colormap != NULL)
				cinfo.colormap = (JSAMPARRAY)cmap;
		}

		if (overwrite_default(flags, PROGRESSIVE_MODE)) {
			cinfo.progressive_mode = flag_to_boolean_value(flags, PROGRESSIVE_MODE);
		}
		if (overwrite_default(flags, ARITH_CODE)) {
			cinfo.arith_code = flag_to_boolean_value(flags, ARITH_CODE);
		}
		if (overwrite_default(flags, CCIR601_SAMPLING)) {
			cinfo.CCIR601_sampling = flag_to_boolean_value(flags, CCIR601_SAMPLING);
		}
		if (overwrite_default(flags, TWO_PASS_QUANTIZE)) {
			cinfo.two_pass_quantize = flag_to_boolean_value(flags, TWO_PASS_QUANTIZE);
		}
		if (overwrite_default(flags, ENABLE_1PASS_QUANT)) {
			cinfo.enable_1pass_quant = flag_to_boolean_value(flags, ENABLE_1PASS_QUANT);
		}
		if (overwrite_default(flags, ENABLE_EXTERNAL_QUANT)) {
			cinfo.enable_external_quant = flag_to_boolean_value(flags, ENABLE_EXTERNAL_QUANT);
		}
		if (overwrite_default(flags, ENABLE_2PASS_QUANT)) {
			cinfo.enable_2pass_quant = flag_to_boolean_value(flags, ENABLE_2PASS_QUANT);
		}
		// decompress
		(void)jpeg_start_decompress(&cinfo);
		// read pixels
		char *rowptr = (char *)rgb;
		unsigned short stride = cinfo.out_color_components;
		if(overwrite_default(flags, QUANTIZE_COLORS) && flag_to_boolean_value(flags, QUANTIZE_COLORS))
			stride = 1;

		// fprintf(stderr,
		// 	"- JPEG_REACHED_SOS %d\n"
		// 	"- JPEG_REACHED_EOI %d\n"
		// 	"- JPEG_ROW_COMPLETED %d\n"
		// 	"- JPEG_SCAN_COMPLETED %d\n"
		// 	"- JPEG_SUSPENDED %d\n",
		// 	JPEG_REACHED_SOS, JPEG_REACHED_EOI, JPEG_ROW_COMPLETED, JPEG_SCAN_COMPLETED, JPEG_SUSPENDED
		// );
		// do {
		// 	fprintf(stderr, "Scan %d\n", cinfo.input_scan_number);
		// 	// int res_in = jpeg_consume_input(&cinfo);
		// 	// fprintf(stderr, "- res_in %d\n", res_in);
		// 	// JPEG_REACHED_SOS, JPEG_REACHED_EOI, JPEG_ROW_COMPLETED, JPEG_SCAN_COMPLETED, JPEG_SUSPENDED
		// 	jpeg_start_output(&cinfo, cinfo.input_scan_number);
		while (cinfo.output_scanline < cinfo.output_height) {
			jpeg_read_scanlines(
				&cinfo,
				(JSAMPARRAY)&rowptr,
				1
			);
			rowptr += cinfo.output_width * stride;
		}
		// 	jpeg_finish_output(&cinfo);

		// } while(!jpeg_input_complete(&cinfo));



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
    int num_scans,
	int *scan_script,
    short *huffman_bits,
    short *huffman_values,
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
			chroma_factor[0] = samp_factor[0];
			chroma_factor[1] = samp_factor[1];
			for (int ch = 0; ch < cinfo.num_components; ch++) {
				cinfo.comp_info[ch].v_samp_factor = samp_factor[2*ch + 0];
				cinfo.comp_info[ch].h_samp_factor = samp_factor[2*ch + 1];
			}
		} else {
			chroma_factor[0] = cinfo.comp_info[0].v_samp_factor;
			chroma_factor[1] = cinfo.comp_info[0].h_samp_factor;
		}

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
			if (overwrite_default(flags, FORCE_BASELINE))
				force_baseline = flag_to_boolean_value(flags, FORCE_BASELINE);
			jpeg_set_quality(&cinfo, quality, force_baseline);
		}

		// huffman tables
		_write_huff(&cinfo, huffman_bits, huffman_values, quant_tbl_no);

		if (smoothing_factor >= 0) {
			cinfo.smoothing_factor = smoothing_factor;
		}
		// if (in_color_space >= 0)
		//   cinfo.in_color_space = in_color_space;

		#if LIBVERSION >= 70
		if (overwrite_default(flags, DO_FANCY_UPSAMPLING)) {
			cinfo.do_fancy_downsampling = flag_to_boolean_value(flags, DO_FANCY_UPSAMPLING);
		}
		#endif
		if (overwrite_default(flags, PROGRESSIVE_MODE)) {
			cinfo.progressive_mode = flag_to_boolean_value(flags, PROGRESSIVE_MODE);
		}
		if (overwrite_default(flags, PROGRESSIVE_MODE) && flag_to_boolean_value(flags, PROGRESSIVE_MODE)) {
			if(scan_script == NULL) {
				jpeg_simple_progression(&cinfo);
			} else {
				if (cinfo.script_space == NULL || cinfo.script_space_size < num_scans) {
					cinfo.script_space = (jpeg_scan_info *)(*cinfo.mem->alloc_small)(
						(j_common_ptr)&cinfo,
						JPOOL_PERMANENT,
						num_scans * sizeof(jpeg_scan_info)
					);
				}
				cinfo.scan_info = cinfo.script_space;
				cinfo.num_scans = num_scans;

				jpeg_scan_info * scanptr = cinfo.script_space;
				for(int s = 0; s < num_scans; s++) {
					// fprintf(stderr, "Scan %d:\n", s);
					scanptr->comps_in_scan = scan_script[s*17 + 0];
					// fprintf(stderr, "- comps_in_scan %d\n", scanptr->comps_in_scan);
					scanptr->Ss = scan_script[s*17 + 1];
					scanptr->Se = scan_script[s*17 + 2];
					scanptr->Ah = scan_script[s*17 + 3];
					scanptr->Al = scan_script[s*17 + 4];
					for(int ch = 0; ch < 4; ch++) {
						if(scan_script[s*17 + ch + 5] != -1)
							scanptr->component_index[ch] = scan_script[s*17 + ch + 5];
						// fprintf(stderr, "- component_index[%d] %d %d\n", ch, scan_script[s*17 + ch + 5], scanptr->component_index[ch]);
							// TODO: copy ac_tbl_no and dc_tbl_no
					}
					scanptr++;
				}

			}

		}
		if (overwrite_default(flags, OPTIMIZE_CODING)) {
			cinfo.optimize_coding = flag_to_boolean_value(flags, OPTIMIZE_CODING);
		}
		#ifdef C_ARITH_CODING_SUPPORTED
		if (overwrite_default(flags, ARITH_CODE)) {
			cinfo.arith_code = flag_to_boolean_value(flags, ARITH_CODE);
		}
		#endif
		if (overwrite_default(flags, WRITE_JFIF_HEADER)) {
			cinfo.write_JFIF_header = flag_to_boolean_value(flags, WRITE_JFIF_HEADER);
		}
		if (overwrite_default(flags, WRITE_ADOBE_MARKER)) {
			cinfo.write_Adobe_marker = flag_to_boolean_value(flags, WRITE_ADOBE_MARKER);
		}
		if (overwrite_default(flags, CCIR601_SAMPLING)) {
			cinfo.CCIR601_sampling = flag_to_boolean_value(flags, CCIR601_SAMPLING);
		}
		#if LIBVERSION >= 6300
		if(overwrite_default(flags, TRELLIS_QUANT)) {
			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_TRELLIS_QUANT,
				flag_to_boolean_value(flags, TRELLIS_QUANT)
			);
		}
		if(overwrite_default(flags, TRELLIS_QUANT_DC)) {
			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_TRELLIS_QUANT_DC,
				flag_to_boolean_value(flags, TRELLIS_QUANT_DC)
			);
		}
		if(overwrite_default(flags, TRELLIS_Q_OPT)) {
			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_TRELLIS_Q_OPT,
				flag_to_boolean_value(flags, TRELLIS_Q_OPT)
			);
		}
		if(overwrite_default(flags, OVERSHOOT_DERINGING)) {
			printf("Going to overwrite overshoot deringing? %d\n", flag_to_boolean_value(flags, OVERSHOOT_DERINGING));

			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_OVERSHOOT_DERINGING,
				flag_to_boolean_value(flags, OVERSHOOT_DERINGING)
			);
		}

		printf("flags: %llu\n", flags);
		printf("cinfo.master->trellis_quant = %d\n", cinfo.master->trellis_quant);
		printf("cinfo.master->trellis_quant_dc = %d\n", cinfo.master->trellis_quant_dc);
		printf("cinfo.master->master->overshoot_deringing = %d\n", cinfo.master->overshoot_deringing);

		// if(overwrite_default(flags, OPTIMIZE_SCANS)) {
		// 	jpeg_c_set_bool_param(
		// 		&cinfo,
		// 		JBOOLEAN_OPTIMIZE_SCANS,
		// 		flag_to_boolean_value(flags, OPTIMIZE_SCANS)
		// 	);
		// }
		if(overwrite_default(flags, USE_SCANS_IN_TRELLIS)) {
			jpeg_c_set_bool_param(
				&cinfo,
				JBOOLEAN_USE_SCANS_IN_TRELLIS,
				flag_to_boolean_value(flags, USE_SCANS_IN_TRELLIS)
			);
		}
		// JBOOLEAN_TRELLIS_EOB_OPT = 0xD7F73780, /* TRUE=optimize for sequences of EOB */
		// JBOOLEAN_USE_LAMBDA_WEIGHT_TBL = 0x339DB65F, /* TRUE=use lambda weighting table */
		// JBOOLEAN_OVERSHOOT_DERINGING = 0x3F4BBBF9 /* TRUE=preprocess input to reduce ringing of edges on white background */
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