#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

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
	// sanitizing libjpeg errors
	try {

		// allocate
		struct jpeg_decompress_struct cinfo;
		struct jpeg_error_mgr jerr;

		// read jpeg header
		FILE *fp;
		if((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL)
			return 0;

		// set parameters
		if(out_color_space >= 0)
			cinfo.out_color_space = (J_COLOR_SPACE)out_color_space;
		else
			cinfo.out_color_space = (J_COLOR_SPACE)cinfo.jpeg_color_space;
		if(dither_mode >= 0)
			cinfo.dither_mode = (J_DITHER_MODE)dither_mode;
		if(dct_method >= 0)
			cinfo.dct_method = (J_DCT_METHOD)dct_method;


		if (overwrite_flag(flags, DO_FANCY_UPSAMPLING))
			cinfo.do_fancy_upsampling = flag_is_set(flags, DO_FANCY_UPSAMPLING);
		if (overwrite_flag(flags, DO_BLOCK_SMOOTHING))
			cinfo.do_block_smoothing = flag_is_set(flags, DO_BLOCK_SMOOTHING);
		if (overwrite_flag(flags, QUANTIZE_COLORS))
			cinfo.quantize_colors = flag_is_set(flags, QUANTIZE_COLORS);

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

		if (overwrite_flag(flags, PROGRESSIVE_MODE))
			cinfo.progressive_mode = flag_is_set(flags, PROGRESSIVE_MODE);
		if (overwrite_flag(flags, ARITH_CODE))
			cinfo.arith_code = flag_is_set(flags, ARITH_CODE);
		if (overwrite_flag(flags, CCIR601_SAMPLING))
			cinfo.CCIR601_sampling = flag_is_set(flags, CCIR601_SAMPLING);
		if (overwrite_flag(flags, TWO_PASS_QUANTIZE))
			cinfo.two_pass_quantize = flag_is_set(flags, TWO_PASS_QUANTIZE);
		if (overwrite_flag(flags, ENABLE_1PASS_QUANT))
			cinfo.enable_1pass_quant = flag_is_set(flags, ENABLE_1PASS_QUANT);
		if (overwrite_flag(flags, ENABLE_EXTERNAL_QUANT))
			cinfo.enable_external_quant = flag_is_set(flags, ENABLE_EXTERNAL_QUANT);
		if (overwrite_flag(flags, ENABLE_2PASS_QUANT))
			cinfo.enable_2pass_quant = flag_is_set(flags, ENABLE_2PASS_QUANT);

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
		// read quantization colormap
		if (overwrite_flag(flags, QUANTIZE_COLORS) && flag_is_set(flags, QUANTIZE_COLORS)) {
			int N = cinfo.out_color_components;
			for (int ch = 0; ch < N; ch++) {
				for (int i = 0; i < 256; i++) {
					colormap[ch * 256 + i] = cinfo.colormap[ch][i];
					// colormap[ch*256 + i] = cinfo.colormap[i][ch];
				}
			}
		}

		// cleanup
		(void)jpeg_finish_decompress(&cinfo);
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);

		return 1;

	// error handling
	} catch(...) {
		return 0;
	}

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
  short smoothing_factor,
  int num_markers,
  int *marker_types,
  int *marker_lengths,
  unsigned char *markers,
  BITMASK flags
) {
	// sanitizing libjpeg errors
	try {

		// allocate
		struct jpeg_compress_struct cinfo;
		struct jpeg_error_mgr jerr;

		// open the destination file
		FILE *fp;
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
		if (jpeg_color_space != NULL)
			cinfo.in_color_space = (J_COLOR_SPACE)(jpeg_color_space[0]);
		if (num_components != NULL)
			cinfo.input_components = num_components[0];

		jpeg_set_defaults(&cinfo);
		jpeg_set_colorspace(&cinfo, (J_COLOR_SPACE)(jpeg_color_space[1]));

		// set advanced parameters
		if (dct_method >= 0)
			cinfo.dct_method = (J_DCT_METHOD)dct_method;
		int chroma_factor[2];
		if (samp_factor != NULL) {
			chroma_factor[0] = *(samp_factor + 0);
			chroma_factor[1] = *(samp_factor + 1);
			for (int comp = 0; comp < cinfo.num_components; comp++) {
			cinfo.comp_info[comp].h_samp_factor = *(samp_factor + comp * 2 + 0);
			cinfo.comp_info[comp].v_samp_factor = *(samp_factor + comp * 2 + 1);
			}
		} else {
			chroma_factor[0] = cinfo.comp_info[0].h_samp_factor;
			chroma_factor[1] = cinfo.comp_info[0].v_samp_factor;
		}

		// write qt
		if(qt != NULL) {
			_write_qt(&cinfo, qt, quant_tbl_no, 1);
		// write quality
		} else if (quality > 0) {
			// force baseline (8bit quantization)
			boolean force_baseline = FALSE;
			if (overwrite_flag(flags, FORCE_BASELINE))
				force_baseline = flag_is_set(flags, FORCE_BASELINE);
			jpeg_set_quality(&cinfo, quality, force_baseline);
		}

		// if (qt != NULL) {
		//   // add quant_tbl_no association
		//   unsigned qt_u[64];
		//   // component 0
		//   for (int i = 0; i < 64; i++)
		//     qt_u[i] = qt[i]; // (i&7)*8+(i>>3)
		//   jpeg_add_quant_table(&cinfo, 0, qt_u, 100, FALSE);
		//   cinfo.comp_info[0].component_id = 0;
		//   cinfo.comp_info[0].quant_tbl_no = 0;
		//   // component 1
		//   for (int i = 0; i < 64; i++)
		//     qt_u[i] = qt[64 + i]; // (i&7)*8+(i>>3)
		//   jpeg_add_quant_table(&cinfo, 1, qt_u, 100, FALSE);
		//   cinfo.comp_info[1].component_id = 1;
		//   cinfo.comp_info[1].quant_tbl_no = 1;
		//   // component 2
		//   cinfo.comp_info[2].component_id = 2;
		//   cinfo.comp_info[2].quant_tbl_no = 1;
		// } else if (quality >= 0) {
		//   // force baseline (8bit quantization)
		//   bool force_baseline = FALSE;
		//   if (overwrite_flag(flags, FORCE_BASELINE))
		//     force_baseline = flag_is_set(flags, FORCE_BASELINE);
		//   // set quality
		//   jpeg_set_quality(&cinfo, quality, force_baseline);
		// }

		if (smoothing_factor >= 0)
			cinfo.smoothing_factor = smoothing_factor;
		// if (in_color_space >= 0)
		//   cinfo.in_color_space = in_color_space;

		#if JPEG_LIB_VERSION >= 70
		if (overwrite_flag(flags, DO_FANCY_UPSAMPLING)) {
			cinfo.do_fancy_downsampling = flag_is_set(flags, DO_FANCY_UPSAMPLING);
			// cinfo.min_DCT_h_scaled_size = 5;
			// cinfo.min_DCT_v_scaled_size = 5;
		}
		#endif
		if (overwrite_flag(flags, PROGRESSIVE_MODE))
			cinfo.progressive_mode = flag_is_set(flags, PROGRESSIVE_MODE);
		if (overwrite_flag(flags, PROGRESSIVE_MODE) && flag_is_set(flags, PROGRESSIVE_MODE))
			jpeg_simple_progression(&cinfo);
		if (overwrite_flag(flags, OPTIMIZE_CODING))
			cinfo.optimize_coding = flag_is_set(flags, OPTIMIZE_CODING);
		#ifdef C_ARITH_CODING_SUPPORTED
		if (overwrite_flag(flags, ARITH_CODE))
			cinfo.arith_code = flag_is_set(flags, ARITH_CODE);
		#endif
		if (overwrite_flag(flags, WRITE_JFIF_HEADER))
			cinfo.write_JFIF_header = flag_is_set(flags, WRITE_JFIF_HEADER);
		if (overwrite_flag(flags, WRITE_ADOBE_MARKER))
			cinfo.write_Adobe_marker = flag_is_set(flags, WRITE_ADOBE_MARKER);
		if (overwrite_flag(flags, CCIR601_SAMPLING))
			cinfo.CCIR601_sampling = flag_is_set(flags, CCIR601_SAMPLING);

		// fprintf(stderr, "number of scans: %d\n", cinfo.num_scans);
		// fprintf(stderr, "components in scan: %d\n", cinfo.comps_in_scan);
		// fprintf(stderr, "Ss: %d\n", cinfo.Ss);
		// fprintf(stderr, "Se: %d\n", cinfo.Se);
		// fprintf(stderr, "Ah: %d\n", cinfo.Ah);
		// fprintf(stderr, "Al: %d\n", cinfo.Al);
		// fprintf(stderr, " %d\n", cinfo.Al);
		jpeg_start_compress(&cinfo, TRUE);

		// write markers
		int offset = 0;
		for (int i = 0; i < num_markers; i++) {
			jpeg_write_marker(
				&cinfo,
				marker_types[i],
				(const JOCTET *)(markers + offset),
				marker_lengths[i]
			);
			offset += marker_lengths[i];
		}

		// write data
		char *rowptr = (char *)rgb;
		for (unsigned h = 0; h < cinfo.image_height; h++) {
			jpeg_write_scanlines(
				&cinfo,
				(JSAMPARRAY)&rowptr,
				1
			);
			rowptr += cinfo.image_width * cinfo.input_components;
		}
		// cleanup
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);
		fclose(fp);

		return 1;

	// error handling
	} catch(...) {
		return 0;
	}

}


#ifdef __cplusplus
}
#endif