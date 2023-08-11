
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

/**
 * @brief Helper for indexing DCT block base[channel][w][h].
 *
 * @param base Base pointer.
 * @param channel Channel.
 * @param h Block height.
 * @param w Block width.
 * @param Hmax Height of image in blocks.
 * @param Wmax Width of image in blocks.
 * @return void* Pointer to DCT block base[channel][w][h].
 */
void *_dct_offset(
	short **base,
	int channel,
	int h,
	int w,
	int Hmax,
	int Wmax
) {
	return (void *)(base[channel] + 64 * (w + Wmax * (h + Hmax * (0))));
}


int read_jpeg_dct(
  const char *srcfile,
  short *Y,
  short *Cb,
  short *Cr,
  short *K,
  unsigned short *qt,
  short *quant_tbl_no
) {
	// allocate
	FILE *fp;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// read jpeg header
		if((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL) return 0;

		// read DCT
		jvirt_barray_ptr *coeffs_array = jpeg_read_coefficients(&cinfo);
		// read dct
		JBLOCKARRAY buffer_one;
		JCOEFPTR blockptr_one;
		short *dct[4] = {Y, Cb, Cr, K};
		for(int ch = 0; ch < cinfo.num_components; ch++) {
			if(dct[ch] == NULL) continue; // skip component, if null
			jpeg_component_info* compptr_one = cinfo.comp_info + ch;
			int Hblocks = compptr_one->height_in_blocks; // max height
			int Wblocks = compptr_one->width_in_blocks; // max width
			for(int h = 0; h < Hblocks; h++) { // iterate height
			buffer_one = (cinfo.mem->access_virt_barray)((j_common_ptr)&cinfo, coeffs_array[ch], h, (JDIMENSION)1, FALSE);
			for(int w = 0; w < Wblocks; w++) {
				blockptr_one = buffer_one[0][w];
				for(int bh = 0; bh < 8; bh++) {
				for(int bw = 0; bw < 8; bw++) {
					int i = bh*8 + bw;
					((short *)_dct_offset(dct, ch, h, w, Hblocks, Wblocks))[i] = blockptr_one[bh*8 + bw];
				}
				// memcpy(_dct_offset(dct, ch, w, h, WblocksY, HblocksY), (void *)blockptr_one, sizeof(short)*64);
				}
			}
			}
		}

		// read quantization table
		if (qt != NULL) {
			for (int ch = 0; ch < cinfo.num_components; ch++) {
				int qt_i = cinfo.comp_info[ch].quant_tbl_no;
				quant_tbl_no[ch] = qt_i; // copy quantization table assignment
			}

			for (int ch = 0; ch < 4; ch++) {
				if (cinfo.quant_tbl_ptrs[ch] == NULL)
					continue;
				for (int i = 0; i < 64; i++) {
					qt[ch * 64 + i] = cinfo.quant_tbl_ptrs[ch]->quantval[i]; //[(i&7)*8+(i>>3)];
				}
			}
		}

		// cleanup
		jpeg_finish_decompress(&cinfo);

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

/**
 * Writs the DCT coefficients into the destination file.
 *
 * You must specify either srcfile, qt or quality.
 *
 * @param srcfile Source file to copy parameters from or NULL.
 * @param dstfile Destination file to write to.
 * @param dct DCT coefficient matrix of dimensions 3 * W/8 * H/8 * 8 * 8.
 * @param image_dims Image pixel dimensions.
 * @param in_color_space JPEG color space.
 * @param in_components Number of channels.
 * @param samp_factor Sampling factor of shape 2 * 3 or NULL.
 * @param qt Quantization table of shape 2 * 8 * 8 or NULL.
 * @param quality Output quality between 0 - 100. -1, if not used.
 */
int write_jpeg_dct(
	const char *srcfile,
	const char *dstfile,
	short *Y,
	short *Cb,
	short *Cr,
	short *K,
	int *image_dims,
	int *block_dims,
	int *samp_factor,
	int in_color_space,
	int in_components,
	unsigned short *qt,
	short quality,
	short *quant_tbl_no,
	int num_markers,
	int *marker_types,
	int *marker_lengths,
	unsigned char *markers,
	BITMASK flags
) {
	// allocate
	FILE *fp_in = NULL, *fp_out = NULL;
	struct jpeg_compress_struct cinfo_out;
	struct jpeg_decompress_struct cinfo_in;
	struct jpeg_error_mgr jerr_in, jerr_out;
	memset((void *)&cinfo_out, 0x0, sizeof(struct jpeg_compress_struct));
	memset((void *)&jerr_out, 0x0, sizeof(struct jpeg_error_mgr));

	// sanitizing libjpeg errors
	int status = 1;
	try {

		// check inputs
		if (dstfile == NULL) {
			fprintf(stderr, "you must specify dstfile\n");
			return 0;
		}
		if (Y == NULL) {
			fprintf(stderr, "you must specify Y\n");
			return 0;
		}
		if (((Cb != NULL) && (Cr == NULL)) || ((Cb == NULL) && (Cr != NULL))) {
			fprintf(stderr, "you must specify Y or YCbCr\n");
			return 0;
		}

		// open the destination file
		if ((fp_out = fopen(dstfile, "wb")) == NULL) {
			fprintf(stderr, "can't open %s\n", dstfile);
			return 0;
		}

		// read source jpeg
		if (srcfile != NULL) {
			if ((fp_in = _read_jpeg(srcfile, &cinfo_in, &jerr_in, FALSE)) == NULL)
				return 0;
			// todo write markers
			(void)jpeg_read_header(&cinfo_in, TRUE);
		}
		cinfo_out.err = jpeg_std_error(&jerr_out);
		jpeg_create_compress(&cinfo_out);
		jpeg_stdio_dest(&cinfo_out, fp_out);
		// copy critical parameters to dstfile
		if (srcfile != NULL)
			jpeg_copy_critical_parameters((j_decompress_ptr)&cinfo_in, &cinfo_out);

		cinfo_out.image_height = image_dims[0];
		cinfo_out.image_width = image_dims[1];
		cinfo_out.in_color_space = (J_COLOR_SPACE)in_color_space;
		cinfo_out.jpeg_color_space = (J_COLOR_SPACE)in_color_space;
		if (in_components >= 0)
			cinfo_out.input_components = in_components;
		cinfo_out.num_components = cinfo_out.input_components;

		if (srcfile == NULL) // set defaults
			jpeg_set_defaults(&cinfo_out);

		// set sampling factors
		int chroma_factor[2];
		if(samp_factor != NULL) {

			chroma_factor[0] = *(samp_factor + 0);
			chroma_factor[1] = *(samp_factor + 1);
			for(int comp = 0; comp < cinfo_out.num_components; comp++) {
				cinfo_out.comp_info[comp].v_samp_factor = *(samp_factor + comp*2 + 0);
				cinfo_out.comp_info[comp].h_samp_factor = *(samp_factor + comp*2 + 1);
			}
		}

		// write qt
		if(qt != NULL)
			_write_qt(&cinfo_out, qt, quant_tbl_no, 0);
		// write quality
		else if (quality > 0) {
			jpeg_set_quality(&cinfo_out, quality, TRUE);
		}
		if (overwrite_flag(flags, OPTIMIZE_CODING)) {
			cinfo_out.optimize_coding = flag_is_set(flags, OPTIMIZE_CODING);
		}
		#ifdef C_ARITH_CODING_SUPPORTED
		if (overwrite_flag(flags, ARITH_CODE)) {
			cinfo_out.arith_code = flag_is_set(flags, ARITH_CODE);
		}
		#endif

		// DCT coefficients
		jvirt_barray_ptr *coeffs_array;
		if (srcfile != NULL) { // copy from source
			coeffs_array = jpeg_read_coefficients(&cinfo_in);
		} else { // allocate new
			coeffs_array = (jvirt_barray_ptr *)(cinfo_out.mem->alloc_small)(
				(j_common_ptr)&cinfo_out,
				JPOOL_IMAGE,
				sizeof(jvirt_barray_ptr) * cinfo_out.num_components
			);
			for (int ch = 0; ch < (cinfo_out.num_components); ch++) { // channel iterator
				jpeg_component_info *comp_ptr = cinfo_out.comp_info + ch;
				comp_ptr->height_in_blocks = (JDIMENSION)block_dims[2 * ch];
				comp_ptr->width_in_blocks = (JDIMENSION)block_dims[2 * ch + 1];
				coeffs_array[ch] = (cinfo_out.mem->request_virt_barray)(
					(j_common_ptr)&cinfo_out,
					JPOOL_IMAGE,
					TRUE,
					(JDIMENSION)jround_up(comp_ptr->width_in_blocks, // component size in dct blocks (ignoring mcu)
										comp_ptr->h_samp_factor),  // round up is important, if border MCUs are not completely needed
					(JDIMENSION)jround_up(comp_ptr->height_in_blocks,
										comp_ptr->v_samp_factor),
					(JDIMENSION)comp_ptr->v_samp_factor
				);
			}
		}

		#if JPEG_LIB_VERSION >= 80
		jpeg_calc_jpeg_dimensions(&cinfo_out);
		#endif

		jpeg_write_coefficients(&cinfo_out, coeffs_array);

		// write markers
		int offset = 0;
		for (int i = 0; i < num_markers; i++) {
			jpeg_write_marker(
				&cinfo_out,
				marker_types[i],
				(const JOCTET *)(markers + offset),
				marker_lengths[i]
			);
			offset += marker_lengths[i];
		}

		// write DCT coefficients
		JBLOCKARRAY buffer_one;
		JCOEFPTR blockptr_one;
		short *dct[4] = {Y, Cb, Cr, K};
		for (int ch = 0; ch < 3; ch++) { // channel iterator
			if (dct[ch] == NULL) continue;
			jpeg_component_info *comp_ptr = cinfo_out.comp_info + ch;
			int Hblocks = comp_ptr->height_in_blocks; // max height
			int Wblocks = comp_ptr->width_in_blocks;  // max width
			for (int h = 0; h < Hblocks; h++) { // height iterator
				buffer_one = (cinfo_out.mem->access_virt_barray)((j_common_ptr)&cinfo_out, coeffs_array[ch], h, (JDIMENSION)1, TRUE);
				for (int w = 0; w < Wblocks; w++) { // width iterator
					blockptr_one = buffer_one[0][w];
					for (int bh = 0; bh < 8; bh++)
						for (int bw = 0; bw < 8; bw++)
							blockptr_one[bh * 8 + bw] = ((short *)_dct_offset(dct, ch, h, w, Hblocks, Wblocks))[bh * 8 + bw];
				}
			}
		}

		// finish
		jpeg_finish_compress(&cinfo_out);
		if (srcfile != NULL)
			jpeg_finish_decompress(&cinfo_in);

	// error handling
	} catch(...) {
		status = 0;
	}

	// cleanup and return
	jpeg_destroy_compress(&cinfo_out);
	if(fp_out != NULL)
		fclose(fp_out);
	if (srcfile != NULL) {
		jpeg_destroy_decompress(&cinfo_in);
		if(fp_in != NULL)
			fclose(fp_in);
	}
	return status;

}


#ifdef __cplusplus
}
#endif