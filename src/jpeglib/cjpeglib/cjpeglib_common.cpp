#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib_common.h"

extern int gpos;
extern int gmarker_types[MAX_MARKER];
extern int gmarker_lengths[MAX_MARKER];
extern unsigned char * gmarker_data[MAX_MARKER];

void my_custom_error_exit(
	j_common_ptr cinfo
) {
	// Write the message
	(*cinfo->err->output_message)(cinfo);

	// Let the memory manager delete any temp files before we die
	jpeg_destroy(cinfo);

	// Throw an "exception" using C++
	throw 0;
}

FILE *_read_jpeg(const char *filename,
                 struct jpeg_decompress_struct *cinfo,
                 struct jpeg_error_mgr *jerr,
                 bool read_header) {
	// open file
	FILE *fp;
	if ((fp = fopen(filename, "rb")) == NULL) {
		fprintf(stderr, "not possible to open %s\n", filename);
		return NULL;
	}

	// check file size
	fseek(fp, 0L, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
	if(fsize == 0) return NULL;

	// zero the structures
	memset(cinfo,0x00,sizeof(struct jpeg_decompress_struct));
	memset(jerr,0x00,sizeof(struct jpeg_error_mgr));

	//load image
	cinfo->err = jpeg_std_error(jerr);
	jpeg_create_decompress(cinfo);
	jpeg_stdio_src(cinfo, fp);

	if(read_header)
		(void) jpeg_read_header(cinfo, TRUE);

	return fp;
}

int read_jpeg_info(
	const char *srcfile,
	int *block_dims,
	int *image_dims,
	int *num_components,
	int *samp_factor,
	int *jpeg_color_space,
	int *marker_lengths,
	int *marker_types,
    short *huffman_bits,
    short *huffman_values,
	int *num_scans,
	BITMASK *flags
) {
	// allocate
	FILE *fp = NULL;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// read jpeg header
		if((fp = _read_jpeg(srcfile, &cinfo, &jerr, FALSE)) == NULL) return 0;

		// markers
		if((marker_lengths != NULL) || (marker_types != NULL)) {
			// setup
			set_marker_handlers(&cinfo);
			// read markers
			(void) jpeg_read_header(&cinfo, TRUE);
			// collect marker data
			for(int i = 0; i < gpos; i++) {
				marker_lengths[i] = gmarker_lengths[i];
				marker_types[i] = gmarker_types[i];
			}
			unset_marker_handlers(&cinfo);
		} else {
			(void) jpeg_read_header(&cinfo, TRUE);
		}

		jpeg_calc_output_dimensions(&cinfo);

		cinfo.buffered_image = TRUE;
		(void)jpeg_start_decompress(&cinfo);
		// (void)jpeg_read_coefficients(&cinfo);

		// copy to caller
		if (block_dims != NULL) {
			for (int ch = 0; ch < cinfo.num_components; ch++) {
				block_dims[2*ch] = cinfo.comp_info[ch].height_in_blocks;
				block_dims[2*ch + 1] = cinfo.comp_info[ch].width_in_blocks;
			}
		}
		if (image_dims != NULL) {
			image_dims[0] = cinfo.output_height;
			image_dims[1] = cinfo.output_width;
		}
		if (num_components != NULL) {
			num_components[0] = cinfo.num_components;
			num_components[1] = cinfo.out_color_components;
			num_components[2] = cinfo.output_components;
		}
		if (jpeg_color_space != NULL) {
			jpeg_color_space[0] = cinfo.out_color_space;
			jpeg_color_space[1] = cinfo.jpeg_color_space;
		}

		if (samp_factor != NULL)
			for (int ch = 0; ch < cinfo.num_components; ch++) {
				samp_factor[ch*2 + 0] = cinfo.comp_info[ch].v_samp_factor;
				samp_factor[ch*2 + 1] = cinfo.comp_info[ch].h_samp_factor;
			}

		if (flags != NULL) {
			*flags = (((cinfo.progressive_mode) ? (-1) : 0) & PROGRESSIVE_MODE) | (*flags);
		}

		if(huffman_bits != NULL) {
			for(int ch = 0; ch < 4; ch++) {
				// *(huffman_valid + (ch)) = cinfo.dc_huff_tbl_ptrs[ch] != NULL;
				// *(huffman_valid + (ch + 4)) = cinfo.ac_huff_tbl_ptrs[ch] != NULL;
				if(cinfo.dc_huff_tbl_ptrs[ch] == NULL)
					huffman_bits[(0*4 + ch) * 17] = -1;
				if(cinfo.ac_huff_tbl_ptrs[ch] == NULL)
					huffman_bits[(1*4 + ch) * 17] = -1;

				// huffman tables - bits
				int dc_max = 0, ac_max = 0;
				for(int i = 0; i < 17; i++) {
					if(cinfo.dc_huff_tbl_ptrs[ch] != NULL) {
						huffman_bits[(0*4 + ch) * 17 + i] = cinfo.dc_huff_tbl_ptrs[ch]->bits[i];
						dc_max += cinfo.dc_huff_tbl_ptrs[ch]->bits[i];
					}
					if(cinfo.ac_huff_tbl_ptrs[ch] != NULL) {
						huffman_bits[(1*4 + ch) * 17 + i] = cinfo.ac_huff_tbl_ptrs[ch]->bits[i];
						ac_max += cinfo.ac_huff_tbl_ptrs[ch]->bits[i];
					}
				}
				// hufman tables - values
				for(int v = 0; v < 256; v++) {
					if(cinfo.dc_huff_tbl_ptrs[ch] != NULL && v < dc_max)
						huffman_values[(0*4 + ch) * 256 + v] = cinfo.dc_huff_tbl_ptrs[ch]->huffval[v];
					if(cinfo.ac_huff_tbl_ptrs[ch] != NULL && v < ac_max)
						huffman_values[(1*4 + ch) * 256 + v] = cinfo.ac_huff_tbl_ptrs[ch]->huffval[v];
				}
			}
		}

		// load number of scans (reads image data)
		*num_scans = 0;
		do {
			jpeg_start_output(&cinfo, cinfo.input_scan_number);
			++*num_scans;
			jpeg_finish_output(&cinfo);
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

void _write_qt(
	struct jpeg_compress_struct * cinfo,
	unsigned short *qt,
	short *quant_tbl_no,
	unsigned char only_create
) {
	if (qt != NULL) {
		unsigned qt_u[64];
		unsigned char qt_slot_seen = 0;
		for(int ch = 0; ch < cinfo->num_components; ch++) {
			// get qt slot for component
			int qt_ch = ch;
			if(quant_tbl_no != NULL) {
				qt_ch = quant_tbl_no[ch];
				if(qt_ch < 0)
					continue;
			}
			// fprintf(stderr, "Channel %d:\n", ch);
			cinfo->comp_info[ch].component_id = ch;
			cinfo->comp_info[ch].quant_tbl_no = qt_ch;
			if(!(qt_slot_seen & (0x1 << qt_ch))) {
				qt_slot_seen |= 0x1 << qt_ch;

				// allocate slot
				if(only_create || (cinfo->quant_tbl_ptrs[qt_ch] == NULL)) {
					for (int i = 0; i < 64; i++)
						qt_u[i] = qt[ch * 64 + i];
					jpeg_add_quant_table(cinfo, qt_ch, qt_u, 100, FALSE);

				// just change
				} else {
					// fprintf(stderr, " - change slot\n");
					for (int i = 0; i < 64; i++)
						cinfo->quant_tbl_ptrs[qt_ch]->quantval[i] = qt[qt_ch * 64 + i];
				}
			}

		}
	}
}

void _write_huff(
	struct jpeg_compress_struct *cinfo,
	short *huffman_bits,
	short *huffman_values,
	short *quant_tbl_no
) {
	unsigned char slot_seen = 0;
	for(int ch = 0; ch < 4; ch++) {
		int qt_ch = ch;
		if(quant_tbl_no != NULL) {
			qt_ch = quant_tbl_no[ch];
			if(qt_ch < 0)
				continue;
		}
		if(!(slot_seen & (0x1 << qt_ch))) {
			slot_seen |= 0x1 << qt_ch;

			// create standard huffmans
			if(huffman_bits == NULL) {
				// fprintf(stderr, "create standard huffman\n");
				#if (LIBVERSION >= 94) && (LIBVERSION < 3000) // =>9d
				// if(!cinfo->ac_huff_tbl_ptrs[qt_ch]) {
				// 	jpeg_std_huff_table((j_common_ptr)cinfo, FALSE, qt_ch);
				// 	fprintf(stderr, "- AC\n");
				// }
				// if(!cinfo->dc_huff_tbl_ptrs[qt_ch]) {
				// 	jpeg_std_huff_table((j_common_ptr)cinfo, TRUE, qt_ch);
				// 	fprintf(stderr, "- DC\n");
				// }
				#endif

			// add custom huffmans
			} else {
				// fprintf(stderr, "set custom huffman to %d\n", ch);
				// DC
				if(huffman_bits[(0*4 + ch) * 17] != -1) {
					if(!cinfo->dc_huff_tbl_ptrs[qt_ch])
						cinfo->dc_huff_tbl_ptrs[qt_ch] = jpeg_alloc_huff_table((j_common_ptr)cinfo);
					for(int i = 0; i < 17; i++) {
						cinfo->dc_huff_tbl_ptrs[ch]->bits[i] = huffman_bits[(0*4 + ch) * 17 + i];
					}
					for(int v = 0; v < 256; v++) {
						cinfo->dc_huff_tbl_ptrs[ch]->huffval[v] = huffman_values[(0*4 + ch) * 256 + v];
					}
				}
				// AC
				if(huffman_bits[(1*4 + ch) * 17] != -1) {
					if(!cinfo->ac_huff_tbl_ptrs[qt_ch])
						cinfo->ac_huff_tbl_ptrs[qt_ch] = jpeg_alloc_huff_table((j_common_ptr)cinfo);
					for(int i = 0; i < 17; i++) {
						cinfo->ac_huff_tbl_ptrs[ch]->bits[i] = huffman_bits[(1*4 + ch) * 17 + i];
					}
					for(int v = 0; v < 256; v++) {
						cinfo->ac_huff_tbl_ptrs[ch]->huffval[v] = huffman_values[(1*4 + ch) * 256 + v];
					}
				}
			}


		}

	}
}

int print_jpeg_params(
	const char *srcfile
) {

	// allocate
	FILE *fp = NULL;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// read jpeg header
		if ((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL) return 0;
		(void)jpeg_start_decompress(&cinfo);

		printf("File %s details:\n", srcfile);
		printf("  image_height             %d\n", cinfo.image_height);
		printf("  image_width              %d\n", cinfo.image_width);
		printf("  jpeg_color_space         %d\n", cinfo.jpeg_color_space);
		printf("  out_color_space          %d\n", cinfo.out_color_space);
		printf("  scale_num                %u\n", cinfo.scale_num);
		printf("  scale_denom              %u\n", cinfo.scale_denom);
		printf("  output_gamma             %f\n", cinfo.output_gamma);
		printf("  buffered_image           %d\n", cinfo.buffered_image);
		printf("  dct_method               %d\n", cinfo.dct_method);
		printf("  do_fancy_upsampling      %d\n", cinfo.do_fancy_upsampling);
		printf("  do_block_smoothing       %d\n", cinfo.do_block_smoothing);
		printf("  quantize_colors          %d\n", cinfo.quantize_colors);
		printf("  dither_mode              %d\n", cinfo.dither_mode);
		printf("  two_pass_quantize        %d\n", cinfo.two_pass_quantize);
		printf("  desired_number_of_colors %d\n", cinfo.desired_number_of_colors);
		printf("  enable_1pass_quant       %d\n", cinfo.enable_1pass_quant);
		printf("  enable_external_quant    %d\n", cinfo.enable_external_quant);
		printf("  enable_2pass_quant       %d\n", cinfo.enable_2pass_quant);
		printf("  output_width             %d\n", cinfo.output_width);
		printf("  output_height            %d\n", cinfo.output_height);
		printf("  out_color_components     %d\n", cinfo.out_color_components);
		printf("  output_components        %d\n", cinfo.output_components);
		printf("  actual_number_of_colors  %d\n", cinfo.actual_number_of_colors);
		printf("  output_scanline          %d\n", cinfo.output_scanline);
		printf("  input_scan_number        %d\n", cinfo.input_scan_number);
		printf("  input_iMCU_row           %d\n", cinfo.input_iMCU_row);
		printf("  output_scan_number       %d\n", cinfo.output_scan_number);
		printf("  output_iMCU_row          %d\n", cinfo.output_iMCU_row);
		printf("  data_precision           %d\n", cinfo.data_precision);
		printf("  progressive_mode         %d\n", cinfo.progressive_mode);
		printf("  arith_code               %d\n", cinfo.arith_code);

		for (int comp = 0; comp < cinfo.out_color_components; comp++) {
			printf("  comp_info[%d]\n", comp);
			printf("    h_samp_factor          %d\n", cinfo.comp_info[comp].h_samp_factor);
			printf("    v_samp_factor          %d\n", cinfo.comp_info[comp].v_samp_factor);
			printf("    quant_tbl_no           %d\n", cinfo.comp_info[comp].quant_tbl_no);
			printf("    dc_tbl_no              %d\n", cinfo.comp_info[comp].dc_tbl_no);
			printf("    ac_tbl_no              %d\n", cinfo.comp_info[comp].ac_tbl_no);
			printf("    width_in_blocks        %d\n", cinfo.comp_info[comp].width_in_blocks);
			printf("    height_in_blocks       %d\n", cinfo.comp_info[comp].height_in_blocks);
			printf("    downsampled_width      %d\n", cinfo.comp_info[comp].downsampled_width);
			printf("    downsampled_height     %d\n", cinfo.comp_info[comp].downsampled_height);
			printf("    component_needed       %d\n", cinfo.comp_info[comp].component_needed);
			printf("    MCU_width              %d\n", cinfo.comp_info[comp].MCU_width);
			printf("    MCU_height             %d\n", cinfo.comp_info[comp].MCU_height);
			printf("    MCU_blocks             %d\n", cinfo.comp_info[comp].MCU_blocks);
			printf("    MCU_sample_width       %d\n", cinfo.comp_info[comp].MCU_sample_width);
			printf("    last_col_width         %d\n", cinfo.comp_info[comp].last_col_width);
			printf("    last_row_height        %d\n", cinfo.comp_info[comp].last_row_height);
		}

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
