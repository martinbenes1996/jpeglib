#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib_common.h"

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
		fprintf(stderr, "not possible to open\n");
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
  BITMASK *flags
) {
	// sanitizing libjpeg errors
	try {

		// allocate
		struct jpeg_decompress_struct cinfo;
		struct jpeg_error_mgr jerr;

		// read jpeg header
		FILE *fp;
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

		(void)jpeg_read_coefficients(&cinfo);

		// copy to caller
		if (block_dims != NULL) {
			for (int i = 0; i < cinfo.num_components; i++) {
			block_dims[2 * i] = cinfo.comp_info[i].height_in_blocks;
			block_dims[2 * i + 1] = cinfo.comp_info[i].width_in_blocks;
			}
		}
		if (image_dims != NULL) {
			image_dims[0] = cinfo.output_height;
			image_dims[1] = cinfo.output_width;
		}
		if (num_components != NULL) {
			num_components[0] = cinfo.num_components;
			// num_components[1] = cinfo.out_color_components;
			// num_components[2] = cinfo.output_components;
		}
		if (jpeg_color_space != NULL) {
			jpeg_color_space[0] = cinfo.out_color_space;
			jpeg_color_space[1] = cinfo.jpeg_color_space;
		}

		if (samp_factor != NULL)
			for (int comp = 0; comp < cinfo.num_components; comp++) {
			*(samp_factor + comp * 2 + 0) = cinfo.comp_info[comp].h_samp_factor;
			*(samp_factor + comp * 2 + 1) = cinfo.comp_info[comp].v_samp_factor;
			}

		if (flags != NULL) {
			(*flags) = (((cinfo.progressive_mode) ? (-1) : 0) & PROGRESSIVE_MODE) | (*flags);
		}

		// cleanup
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);

		return 1;

	// error handling
	} catch(...) {
		return 0;
	}

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
		for(int ch = 0; ch < 4; ch++) {
			// get qt slot for component
			int qt_ch = ch;
			if(quant_tbl_no != NULL) {
				qt_ch = quant_tbl_no[ch];
				if(qt_ch < 0)
				continue;
			}
			cinfo->comp_info[ch].component_id = ch;
			cinfo->comp_info[ch].quant_tbl_no = qt_ch;
			if(!(qt_slot_seen & (0x1 << qt_ch))) {
				qt_slot_seen |= 0x1 << qt_ch;

				// allocate slot
				if(only_create || (cinfo->quant_tbl_ptrs[qt_ch] == NULL)) {

				for (int i = 0; i < 64; i++)
					qt_u[i] = qt[ch * 64 + i];
				jpeg_add_quant_table(cinfo, qt_ch, qt_u, 100, FALSE);

				if(!cinfo->ac_huff_tbl_ptrs[qt_ch]) {
					int i = qt_ch;
					while((i >= 0) && !cinfo->ac_huff_tbl_ptrs[--i]) ;
					cinfo->ac_huff_tbl_ptrs[qt_ch] = jpeg_alloc_huff_table((j_common_ptr)cinfo);
					memcpy(cinfo->ac_huff_tbl_ptrs[qt_ch], cinfo->ac_huff_tbl_ptrs[i], sizeof(JHUFF_TBL));
				}
				if(!cinfo->dc_huff_tbl_ptrs[qt_ch]) {
					int i = qt_ch;
					while((i >= 0) && !cinfo->dc_huff_tbl_ptrs[--i]) {}
					cinfo->dc_huff_tbl_ptrs[qt_ch] = jpeg_alloc_huff_table((j_common_ptr)cinfo);
					memcpy(cinfo->dc_huff_tbl_ptrs[qt_ch], cinfo->dc_huff_tbl_ptrs[i], sizeof(JHUFF_TBL));
				}

				// just change
				} else {
				for (int i = 0; i < 64; i++)
					cinfo->quant_tbl_ptrs[qt_ch]->quantval[i] = qt[qt_ch * 64 + i];

				}
			}
		}
	}
}


int print_jpeg_params(
	const char *srcfile
) {
	// sanitizing libjpeg errors
	try {

		// allocate
		struct jpeg_decompress_struct cinfo;
		struct jpeg_error_mgr jerr;

		// read jpeg header
		FILE *fp;
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

		return 1;

	// error handling
	} catch(...) {
		return 0;
	}
}

#ifdef __cplusplus
}
#endif
