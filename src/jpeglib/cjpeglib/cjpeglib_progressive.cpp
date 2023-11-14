#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib_common.h"

int read_jpeg_progressive(
    const char *srcfile,
	unsigned char *rgb,
	unsigned char *colormap, // colormap used
	unsigned char *in_colormap, // colormap to use
	int out_color_space,
	int dither_mode,
	int dct_method,
	int *scan_script,
    short *huffman_bits,
    short *huffman_values,
	unsigned short *qt,
    short *quant_tbl_no,
	BITMASK flags
) {
    // allocate
	FILE *fp = NULL;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

    // sanitizing libjpeg errors
	int status = 1;
	try {
		// read jpeg header
		if((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL) return 0;

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

        jpeg_calc_output_dimensions(&cinfo);

        cinfo.buffered_image = TRUE;
		(void)jpeg_start_decompress(&cinfo);
		// read pixels
		char *rowptr = (char *)rgb;
		unsigned short stride = cinfo.out_color_components;

        int s = 0;
        do {
			// fprintf(stderr, "Scan %d\n", cinfo.input_scan_number);
			jpeg_start_output(&cinfo, cinfo.input_scan_number);

			// quantization tables
			// fprintf(stderr, "Scan %d\n", cinfo.input_scan_number);
			for (int ch = 0; ch < cinfo.comps_in_scan; ch++) {
				int qt_i = cinfo.cur_comp_info[ch]->quant_tbl_no;
				// fprintf(stderr, " - component %d: quant_tbl_no %d\n", ch, qt_i);
				quant_tbl_no[s * 4 + ch] = qt_i;
				if (cinfo.quant_tbl_ptrs[ch] == NULL)
					continue;
				for (int i = 0; i < 64; i++) {
					// fprintf(stderr, "%d ", cinfo.quant_tbl_ptrs[ch]->quantval[i]);
					qt[(s * 4 + ch) * 64 + i] = cinfo.quant_tbl_ptrs[ch]->quantval[i];
				}
				// fprintf(stderr, "\n");
			}

			// scan script
			scan_script[s*17 + 0] = cinfo.comps_in_scan;
			scan_script[s*17 + 1] = cinfo.Ss;
			scan_script[s*17 + 2] = cinfo.Se;
			scan_script[s*17 + 3] = cinfo.Ah;
			scan_script[s*17 + 4] = cinfo.Al;
			for(int ch = 0; ch < 4; ch++) {
				if(cinfo.cur_comp_info[ch] != NULL) {
					scan_script[s*17 + ch + 5] = cinfo.cur_comp_info[ch]->component_index;
					scan_script[s*17 + ch + 9] = cinfo.cur_comp_info[ch]->dc_tbl_no;
					scan_script[s*17 + ch + 13] = cinfo.cur_comp_info[ch]->ac_tbl_no;
				} else {
					scan_script[s*17 + ch + 5] = scan_script[s*17 + ch + 9] = scan_script[s*17 + ch + 13] = -1;
				}
			}

			// huffman
			for(int comp = 0; comp < 4; comp++) {
				// huffman_valid[(s * 2 + 0) * 4 + comp] = cinfo.dc_huff_tbl_ptrs[comp] != NULL;
				// huffman_valid[(s * 2 + 1) * 4 + comp] = cinfo.ac_huff_tbl_ptrs[comp] != NULL;
				// huffman tables - bits
				int dc_max = 0, ac_max = 0;
				for(int i = 0; i < 17; i++) {
					if(cinfo.dc_huff_tbl_ptrs[comp] != NULL) {
						huffman_bits[((s * 2 + 0) * 4 + comp) * 17 + i] = cinfo.dc_huff_tbl_ptrs[comp]->bits[i];
						dc_max += cinfo.dc_huff_tbl_ptrs[comp]->bits[i];
					}
					if(cinfo.ac_huff_tbl_ptrs[comp] != NULL) {
                        huffman_bits[((s * 2 + 1) * 4 + comp) * 17 + i] = cinfo.ac_huff_tbl_ptrs[comp]->bits[i];
						ac_max += cinfo.ac_huff_tbl_ptrs[comp]->bits[i];
					}
				}
				// hufman tables - values
				for(int v = 0; v < 256; v++) {
					if(cinfo.dc_huff_tbl_ptrs[comp] != NULL && v < dc_max)
						huffman_values[((s * 2 + 0) * 4 + comp) * 256 + v] = cinfo.dc_huff_tbl_ptrs[comp]->huffval[v];
					if(cinfo.ac_huff_tbl_ptrs[comp] != NULL && v < ac_max)
						huffman_values[((s * 2 + 1) * 4 + comp) * 256 + v] = cinfo.ac_huff_tbl_ptrs[comp]->huffval[v];
				}
			}

			// read spatial
			while (cinfo.output_scanline < cinfo.output_height) {
				jpeg_read_scanlines(
					&cinfo,
					(JSAMPARRAY)&rowptr,
					1
				);
				rowptr += cinfo.output_width * stride;
			}

			jpeg_finish_output(&cinfo);
            s++;
		} while(!jpeg_input_complete(&cinfo));

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

#ifdef __cplusplus
}
#endif
