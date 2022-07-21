#ifdef __cplusplus
extern "C"
{
#endif

// https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpegman.html

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// this is envelope for jpeglib.h
// trying to avoid naming the same as system library
#include "vjpeglib.h"

#include "cjpeglib.h"

// #ifdef USE_TURBO
// #include "jmorecfg.h"
// #endif

// === FLAGS ===
char flag_is_set(BITMASK flags, BITMASK mask) { return (flags & mask) != 0; }
char overwrite_flag(BITMASK flags, BITMASK mask) { return (flags & (mask << 1)) == 0; }
#define DO_FANCY_UPSAMPLING ((BITMASK)0b1 << 0)
#define DO_BLOCK_SMOOTHING ((BITMASK)0b1 << 2)
#define TWO_PASS_QUANTIZE ((BITMASK)0b1 << 4)
#define ENABLE_1PASS_QUANT ((BITMASK)0b1 << 6)
#define ENABLE_EXTERNAL_QUANT ((BITMASK)0b1 << 8)
#define ENABLE_2PASS_QUANT ((BITMASK)0b1 << 10)
#define OPTIMIZE_CODING ((BITMASK)0b1 << 12)
#define PROGRESSIVE_MODE ((BITMASK)0b1 << 14)
#define QUANTIZE_COLORS ((BITMASK)0b1 << 16)
#define ARITH_CODE ((BITMASK)0b1 << 18)
#define WRITE_JFIF_HEADER ((BITMASK)0b1 << 20)
#define WRITE_ADOBE_MARKER ((BITMASK)0b1 << 22)
#define CCIR601_SAMPLING ((BITMASK)0b1 << 24)
#define FORCE_BASELINE ((BITMASK)0b1 << 26)

// === MARKERS ===
// globals
#define MAX_MARKER 20
static int gpos = 0;
static int gmarker_types[MAX_MARKER];
static int gmarker_lengths[MAX_MARKER];
static unsigned char * gmarker_data[MAX_MARKER];
int set_marker_handlers(struct jpeg_decompress_struct * cinfo); // set up handlers
int unset_marker_handlers(struct jpeg_decompress_struct * cinfo); // unset up handlers
int jpeg_getc (j_decompress_ptr cinfo); // read next byte
int jpeg_handle_marker (j_decompress_ptr cinfo); // async marker handler



long jround_up (long a, long b);

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
}

int read_jpeg_markers(
  const char *srcfile,
  unsigned char *markers
) {
  // allocate
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  // read jpeg header
  FILE *fp;
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
  
  //(void)jpeg_read_coefficients(&cinfo);
  
  // cleanup
  jpeg_destroy_decompress(&cinfo);
  fclose(fp);

  return 1;
}

void *_dct_offset(short **base, int channel, int h, int w, int Hmax, int Wmax) {
  return (void *)(base[channel] + 64 * (w + Wmax * (h + Hmax * (0))));
}

int read_jpeg_dct(
  const char *srcfile,
  short *Y,
  short *Cb,
  short *Cr,
  unsigned short *qt,
  unsigned char *quant_tbl_no
) {
  // allocate
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  // read jpeg header
  FILE *fp;
  if((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL) return 0;
  // read DCT
  jvirt_barray_ptr *coeffs_array = jpeg_read_coefficients(&cinfo);
  // read dct
  JBLOCKARRAY buffer_one;
  JCOEFPTR blockptr_one;
  //int HblocksY = cinfo.comp_info->height_in_blocks; // max height
  //int WblocksY = cinfo.comp_info->width_in_blocks; // max width
  short *dct[3] = {Y, Cb, Cr};
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
            int i = bw*8 + bh;
            ((short *)_dct_offset(dct, ch, h, w, Hblocks, Wblocks))[i] = blockptr_one[bh*8 + bw];
          }
          // memcpy(_dct_offset(dct, ch, w, h, WblocksY, HblocksY), (void *)blockptr_one, sizeof(short)*64);
        }
      }
    }
  }

  // read quantization table
  if (qt != NULL) {
    // if(cinfo.num_components > 1) {
    //   if(cinfo.comp_info[1].quant_tbl_no != cinfo.comp_info[2].quant_tbl_no) {
    //     fprintf(stderr, "Mismatching chrominance quantization tables not supported.");
    //     return 0;
    //   }
    // }
    for (int ch = 0; ch < cinfo.num_components; ch++) {
      int qt_i = cinfo.comp_info[ch].quant_tbl_no;
      quant_tbl_no[ch] = qt_i; // copy quantization table assignment
      for (int i = 0; i < 64; i++) {
        qt[ch * 64 + i] = cinfo.quant_tbl_ptrs[qt_i]->quantval[i]; //[(i&7)*8+(i>>3)];
      }
      //(i&7)*8+(i>>3)
      // memcpy((void *)(qt + ch*64), (void *)cinfo.quant_tbl_ptrs[ch]->quantval, sizeof(short)*64);
      // JQUANT_TBL *tbl = cinfo.comp_info[ch].quant_table;
      // memcpy((void *)(qt + ch*64), (void*)tbl->quantval, sizeof(short)*64);
    }
  }


  // cleanup
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(fp);

  return 1;
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
  int *image_dims,
  int *block_dims,
  int in_color_space,
  int in_components,
  unsigned short *qt,
  short quality,
  int num_markers,
  int *marker_types,
  int *marker_lengths,
  unsigned char *markers
) {
  // check inputs
  if (dstfile == NULL) {
    fprintf(stderr, "you must specify dstfile\n");
    return 0;
  }
  // if ((qt == NULL) && (quality < 0)) {
  //   fprintf(stderr, "you must specify either qt or quality\n");
  //   return 0;
  // }
  if (Y == NULL) {
    fprintf(stderr, "you must specify Y\n");
    return 0;
  }
  if (((Cb != NULL) && (Cr == NULL)) || ((Cb == NULL) && (Cr != NULL))) {
    fprintf(stderr, "you must specify Y or YCbCr\n");
    return 0;
  }

  // allocate
  struct jpeg_compress_struct cinfo_out;
  struct jpeg_error_mgr jerr_out;
  // memset((void *)&cinfo_out, 0x0, sizeof(struct jpeg_compress_struct));
  // memset((void *)&jerr_out, 0x0, sizeof(struct jpeg_error_mgr));

  // open the destination file
  FILE *fp_out;
  if ((fp_out = fopen(dstfile, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", dstfile);
    return 0;
  }

  // allocate
  struct jpeg_decompress_struct cinfo_in;
  struct jpeg_error_mgr jerr_in;
  FILE *fp_in;
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

  if (srcfile != NULL) // copy critical parameters to dstfile
    jpeg_copy_critical_parameters((j_decompress_ptr)&cinfo_in, &cinfo_out);

  cinfo_out.image_height = image_dims[0];
  cinfo_out.image_width = image_dims[1];
  cinfo_out.in_color_space = in_color_space;
  cinfo_out.jpeg_color_space = in_color_space;
  if (in_components >= 0)
    cinfo_out.input_components = in_components;
  cinfo_out.num_components = cinfo_out.input_components;

  if (srcfile == NULL) // set defaults
    jpeg_set_defaults(&cinfo_out);

  // set sampling factors
  // int chroma_factor[2];
  // if(samp_factor != NULL) {
  //  chroma_factor[0] = *(samp_factor + 0);
  //  chroma_factor[1] = *(samp_factor + 1);
  // for(int comp = 0; comp < cinfo_out.num_components; comp++) {
  //  cinfo_out.comp_info[comp].h_samp_factor = *(samp_factor + comp*2 + 0);
  //  cinfo_out.comp_info[comp].v_samp_factor = *(samp_factor + comp*2 + 1);
  //}
  //} else {
  //  chroma_factor[0] = cinfo_out.comp_info[0].h_samp_factor;
  //  chroma_factor[1] = cinfo_out.comp_info[0].v_samp_factor;
  //}
  // for(int comp = 0; comp < cinfo.num_components; comp++) {
  //     *(samp_factor + comp*2 + 0) = cinfo.comp_info[comp].h_samp_factor;
  //     *(samp_factor + comp*2 + 1) = cinfo.comp_info[comp].v_samp_factor;
  //   }
  // if(samp_factor != NULL) {
  //   int J_factor = *(samp_factor + 0);
  //   int a_factor = *(samp_factor + 1);
  //   int b_factor = *(samp_factor + 2);

  //   cinfo_out.comp_info[0].h_samp_factor = chroma_factor[0] = J_factor / a_factor;
  //   cinfo_out.comp_info[0].v_samp_factor = chroma_factor[1] = (int)(a_factor == b_factor) + 1;
  //   cinfo_out.comp_info[1].h_samp_factor = cinfo_out.comp_info[1].v_samp_factor = 1;
  //   cinfo_out.comp_info[2].h_samp_factor = cinfo_out.comp_info[2].v_samp_factor = 1;
  // } else {
  //   chroma_factor[0] = cinfo_out.comp_info[0].h_samp_factor;
  //   chroma_factor[1] = cinfo_out.comp_info[0].v_samp_factor;
  // }

  // for(int comp = 0; comp < cinfo_out.input_components; comp++) {
  //   //cinfo_out.comp_info[comp].h_samp_factor
  //   //int J_factor,a_factor,b_factor;
  //   //J,a,b = samp_factor
  //   //  #samp_factor = np.array([
  //   //  #    [int(J / a), int(a == b) + 1],
  //   //fprintf(stderr, "scale %d: %d %d\n", comp, cinfo_out.comp_info[comp].h_samp_factor, cinfo_out.comp_info[comp].v_samp_factor);
  //   cinfo_out.comp_info[comp].h_samp_factor = *(samp_factor + comp*2 + 0);
  //   cinfo_out.comp_info[comp].v_samp_factor = *(samp_factor + comp*2 + 1);
  //   //fprintf(stderr, "scale %d': %d %d\n", comp, cinfo_out.comp_info[comp].h_samp_factor, cinfo_out.comp_info[comp].v_samp_factor);
  // }

  // write qt
  if (qt != NULL)
    for (int ch = 0; ch < 2; ch++)
      for (int i = 0; i < 64; i++)
        cinfo_out.quant_tbl_ptrs[ch]->quantval[i] /*(i&7)*8+(i>>3)*/ = qt[ch * 64 + i];
      // qt[ch*64 + i] = cinfo_out.quant_tbl_ptrs[ch]->quantval[(i&7)*8+(i>>3)];
    // memcpy((void *)cinfo_out.quant_tbl_ptrs[ch]->quantval, (void *)(qt + ch*64), sizeof(short)*64);
  // write quality
  else if (quality > 0)
    jpeg_set_quality(&cinfo_out, quality, TRUE);

  // DCT coefficients
  jvirt_barray_ptr *coeffs_array;
  if (srcfile != NULL) { // copy from source
    coeffs_array = jpeg_read_coefficients(&cinfo_in);
  } else { // allocate new
    coeffs_array = (jvirt_barray_ptr *)(cinfo_out.mem->alloc_small)(
      (j_common_ptr)&cinfo_out,
      JPOOL_IMAGE,
      sizeof(jvirt_barray_ptr) * cinfo_out.num_components);
    for (int ch = 0; ch < (cinfo_out.num_components); ch++) { // channel iterator
      jpeg_component_info *comp_ptr = cinfo_out.comp_info + ch;
      // long v_samp_factor = *(samp_factor + ch*2 + 0);
      // long h_samp_factor = *(samp_factor + ch*2 + 1);
      comp_ptr->height_in_blocks = (JDIMENSION)block_dims[2 * ch];
      comp_ptr->width_in_blocks = (JDIMENSION)block_dims[2 * ch + 1];
      // if(ch > 0) {
      //   comp_ptr->width_in_blocks = ceil(((double)comp_ptr->width_in_blocks) / chroma_factor[0]);
      //   comp_ptr->height_in_blocks = ceil(((double)comp_ptr->height_in_blocks) / chroma_factor[1]);
      // }
      coeffs_array[ch] = (cinfo_out.mem->request_virt_barray)(
        (j_common_ptr)&cinfo_out,
        JPOOL_IMAGE,
        TRUE,
        (JDIMENSION)jround_up(comp_ptr->width_in_blocks, // component size in dct blocks (ignoring mcu)
                              comp_ptr->h_samp_factor),  // round up is important, if border MCUs are not completely needed
        (JDIMENSION)jround_up(comp_ptr->height_in_blocks,
                              comp_ptr->v_samp_factor),
        (JDIMENSION)comp_ptr->v_samp_factor);
    }
  }

  #if JPEG_LIB_VERSION >= 80
  jpeg_calc_jpeg_dimensions(&cinfo_out);
  #endif

  jpeg_write_coefficients(&cinfo_out, coeffs_array);

  // write markers
  int offset = 0;
  for (int i = 0; i < num_markers; i++) {
    jpeg_write_marker(&cinfo_out, marker_types[i], markers + offset, marker_lengths[i]);
    offset += marker_lengths[i];
  }

  // write DCT coefficients
  JBLOCKARRAY buffer_one;
  JCOEFPTR blockptr_one;
  short *dct[3] = {Y, Cb, Cr};
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
            blockptr_one[bh * 8 + bw] = ((short *)_dct_offset(dct, ch, h, w, Hblocks, Wblocks))[bw * 8 + bh];
      }
    }
  }

  // cleanup
  jpeg_finish_compress(&cinfo_out);
  jpeg_destroy_compress(&cinfo_out);
  fclose(fp_out);

  if (srcfile != NULL) {
    jpeg_finish_decompress(&cinfo_in);
    jpeg_destroy_decompress(&cinfo_in);
    fclose(fp_in);
  }

  return 1;
}


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
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  // read jpeg header
  FILE *fp;
  if((fp = _read_jpeg(srcfile, &cinfo, &jerr, TRUE)) == NULL) return 0;

  // set parameters
  if(out_color_space >= 0)
    cinfo.out_color_space = out_color_space;
  else
    cinfo.out_color_space = cinfo.jpeg_color_space;
  if(dither_mode >= 0)
    cinfo.dither_mode = dither_mode;
  if(dct_method >= 0)
    cinfo.dct_method = dct_method;


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
      cinfo.colormap = (char **)cmap;
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
    jpeg_read_scanlines(&cinfo, &rowptr, 1);
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
  short smoothing_factor,
  int num_markers,
  int *marker_types,
  int *marker_lengths,
  unsigned char *markers,
  BITMASK flags
) {
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
    cinfo.in_color_space = jpeg_color_space[0];
  if (num_components != NULL)
    cinfo.input_components = num_components[0];

  jpeg_set_defaults(&cinfo);
  jpeg_set_colorspace(&cinfo, jpeg_color_space[1]);

  // set advanced parameters
  if (dct_method >= 0)
    cinfo.dct_method = dct_method;
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

  if (qt != NULL) {
    unsigned qt_u[64];
    // component 0
    for (int i = 0; i < 64; i++)
      qt_u[i] = qt[i]; // (i&7)*8+(i>>3)
    jpeg_add_quant_table(&cinfo, 0, qt_u, 100, FALSE);
    cinfo.comp_info[0].component_id = 0;
    cinfo.comp_info[0].quant_tbl_no = 0;
    // component 1
    for (int i = 0; i < 64; i++)
      qt_u[i] = qt[64 + i]; // (i&7)*8+(i>>3)
    jpeg_add_quant_table(&cinfo, 1, qt_u, 100, FALSE);
    cinfo.comp_info[1].component_id = 1;
    cinfo.comp_info[1].quant_tbl_no = 1;
    // component 2
    cinfo.comp_info[2].component_id = 2;
    cinfo.comp_info[2].quant_tbl_no = 1;
  } else if (quality >= 0) {
    // force baseline (8bit quantization)
    bool force_baseline = FALSE;
    if (overwrite_flag(flags, FORCE_BASELINE))
      force_baseline = flag_is_set(flags, FORCE_BASELINE);
    // set quality
    jpeg_set_quality(&cinfo, quality, force_baseline);
  }

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

  // start compression
  jpeg_start_compress(&cinfo, TRUE);

  // write markers
  int offset = 0;
  for (int i = 0; i < num_markers; i++) {
    jpeg_write_marker(&cinfo, marker_types[i], markers + offset, marker_lengths[i]);
    offset += marker_lengths[i];
  }

  // write data
  char *rowptr = (char *)rgb;
  for (unsigned h = 0; h < cinfo.image_height; h++) {
    jpeg_write_scanlines(&cinfo, &rowptr, 1);
    rowptr += cinfo.image_width * cinfo.input_components;
  }
  // cleanup
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  fclose(fp);

  return 1;
}

int print_jpeg_params(const char *srcfile) {
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
}

int set_marker_handlers(struct jpeg_decompress_struct *cinfo) {
  // jpeg globals
  gpos = 0;
  for (int i = 0; i < MAX_MARKER; i++) {
    gmarker_types[i] = 0;
    gmarker_data[i] = NULL;
    gmarker_lengths[i] = 0;
  }
  // set handlers
  jpeg_set_marker_processor(cinfo, JPEG_COM, jpeg_handle_marker);
  jpeg_set_marker_processor(cinfo, JPEG_APP0 + 15, jpeg_handle_marker);
  for (int i = 1; i < 14; i++)
    jpeg_set_marker_processor(cinfo, JPEG_APP0 + i, jpeg_handle_marker);

  return 1;
}

int unset_marker_handlers(struct jpeg_decompress_struct *cinfo) {
  // jpeg globals
  for (int i = 0; i < gpos; i++) {
    gmarker_lengths[i] = 0;
    gmarker_types[i] = 0;
    if (gmarker_data[i] != NULL)
      free((void *)gmarker_data[i]);
  }
  gpos = 0;

  // set handlers
  (void)cinfo;

  return 1;
}

int jpeg_getc(j_decompress_ptr cinfo) {
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

int jpeg_handle_marker(j_decompress_ptr cinfo) {
  // get marker name
  char mname[20];
  if (cinfo->unread_marker == JPEG_COM) sprintf(mname, "COM");
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
    if ((p = malloc((length + 1) * sizeof(char))) == NULL) {
      fprintf(stderr, "Bad malloc!\n");
      return FALSE;
    }
    gmarker_data[gpos] = p;
    gpos += 1;

  // too many markers
  } else {
    fprintf(stderr, "Too many markers - %s skipped\n", mname);
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

#ifdef __cplusplus
}
#endif