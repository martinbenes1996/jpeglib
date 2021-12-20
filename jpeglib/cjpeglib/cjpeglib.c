#ifdef __cplusplus
extern "C" {
#endif

// https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpegman.html

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vjpeglib.h"

// #if LIBVERSION == 60
// #include "6b/jpeglib.h"
// #elif LIBVERSION == 80
// #include "8d/jpeglib.h"
// #elif LIBVERSION == 210
// #include "turbo210/jpeglib.h"
// #else
// // intentional syntax error
// Not supported version.
// #endif

//#ifdef USE_TURBO
//#include <jmorecfg.h>
//#endif

#define DO_FANCY_UPSAMPLING 0x1
#define DO_BLOCK_SMOOTHING 0x2
#define TWO_PASS_QUANTIZE 0x4
#define ENABLE_1PASS_QUANT 0x8
#define ENABLE_EXTERNAL_QUANT 0x10
#define ENABLE_2PASS_QUANT 0x20
#define OPTIMIZE_CODING 0x40
#define PROGRESSIVE_MODE 0x80
#define QUANTIZE_COLORS 0x100
#define ARITH_CODE 0x200
#define WRITE_JFIF_HEADER 0x400
#define WRITE_ADOBE_MARKER 0x800
#define CCIR601_SAMPLING 0x1000

GLOBAL(long) jround_up (long a, long b);

// void error_exit (j_common_ptr cinfo)
// {
//   /* Always display the message */
//   (*cinfo->err->output_message) (cinfo);

//   /* Let the memory manager delete any temp files before we die */
//   jpeg_destroy(cinfo);

//   exit(EXIT_FAILURE);
// }

FILE *_read_jpeg(const char *filename,
                 struct jpeg_decompress_struct *cinfo,
                 struct jpeg_error_mgr *jerr) {
  // open file
  FILE *fp;
  if ((fp = fopen(filename, "rb")) == NULL) {
    fprintf(stderr, "can't open %s\n", filename);
    return NULL;
  }

  // zero the structures
  memset(cinfo,0x00,sizeof(struct jpeg_decompress_struct));
  memset(jerr,0x00,sizeof(struct jpeg_error_mgr)); 

  //load image
  cinfo->err = jpeg_std_error(jerr);
  jpeg_create_decompress(cinfo);
  jpeg_stdio_src(cinfo, fp);
  (void) jpeg_read_header(cinfo, TRUE);

  return fp;
}

int read_jpeg_info(
  const char *srcfile,
  int *dct_dims,
  int *image_dims,
  int *num_components,
  int *samp_factor,
  int *jpeg_color_space
) {
  // allocate
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  // read jpeg header
  FILE *fp;
  if((fp = _read_jpeg(srcfile, &cinfo, &jerr)) == NULL) return 0;
  jpeg_calc_output_dimensions(&cinfo);

  (void)jpeg_read_coefficients(&cinfo);

  // copy to caller
  if(dct_dims != NULL) {
    for(int i = 0; i < 3; i++) {
      dct_dims[2*i] = cinfo.comp_info[i].width_in_blocks;
      dct_dims[2*i+1] = cinfo.comp_info[i].height_in_blocks;
    }
  }
  if(image_dims != NULL) {
    image_dims[0] = cinfo.output_width;
    image_dims[1] = cinfo.output_height;
  }
  if(num_components != NULL) {
    num_components[0] = cinfo.num_components;
    //num_components[1] = cinfo.out_color_components;
    //num_components[2] = cinfo.output_components;
  }
  //fprintf(stderr, "setting jpeg color space %d %d\n", cinfo.jpeg_color_space, cinfo.out_color_space);
  if(jpeg_color_space != NULL)
    jpeg_color_space[0] = cinfo.out_color_space;

  if(samp_factor != NULL)
    for(int comp = 0; comp < cinfo.num_components; comp++) {
      *(samp_factor + comp*2 + 0) = cinfo.comp_info[comp].h_samp_factor;
      *(samp_factor + comp*2 + 1) = cinfo.comp_info[comp].v_samp_factor;
    }

  // cleanup
  jpeg_destroy_decompress( &cinfo );
  fclose(fp);

  return 1;
}

void *_dct_offset(short * base, int channel, int w, int h, int Wmax, int Hmax)
{
  return (void *)(base + 64*(h + Hmax*(w + Wmax*(channel))));
}

int read_jpeg_dct(
  const char *srcfile,
  short *dct,
  unsigned short *qt
) { 
  // allocate
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  // read jpeg header
  FILE *fp;
  if((fp = _read_jpeg(srcfile, &cinfo, &jerr)) == NULL) return 0;

  // read DCT
  jvirt_barray_ptr *coeffs_array = jpeg_read_coefficients(&cinfo);
  // read dct
  JBLOCKARRAY buffer_one;
  JCOEFPTR blockptr_one;
  int HblocksY = cinfo.comp_info->height_in_blocks; // max height
  int WblocksY = cinfo.comp_info->width_in_blocks; // max width
  for(int ch = 0; ch < 3; ch++) {
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
            ((short *)_dct_offset(dct, ch, w, h, WblocksY, HblocksY))[i] = blockptr_one[bh*8 + bw];
          }
        }
        //memcpy(_dct_offset(dct, ch, w, h, WblocksY, HblocksY), (void *)blockptr_one, sizeof(short)*64);
      }
    }
  }
  //fprintf(stderr, "Read qt.\n");
  // read quantization table
  if(qt != NULL) {
    for(int ch = 0; ch < 2; ch++) {
      //fprintf(stderr, "qt[%d] = %p -> %p\n", ch, cinfo.quant_tbl_ptrs[ch], cinfo.quant_tbl_ptrs[ch]->quantval);
      for(int i = 0; i < 64; i++) {
        
        qt[ch*64 + i] = cinfo.quant_tbl_ptrs[ch]->quantval[i];//[(i&7)*8+(i>>3)];
      }
      //(i&7)*8+(i>>3)
      //memcpy((void *)(qt + ch*64), (void *)cinfo.quant_tbl_ptrs[ch]->quantval, sizeof(short)*64);
      //JQUANT_TBL *tbl = cinfo.comp_info[ch].quant_table;
      //memcpy((void *)(qt + ch*64), (void*)tbl->quantval, sizeof(short)*64);
    }
  }

  // cleanup
  jpeg_finish_decompress( &cinfo );
  jpeg_destroy_decompress( &cinfo );
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
  short *dct,
  int *image_dims,
  int in_color_space,
  int in_components,
  int *samp_factor,
  unsigned short *qt,
  short quality
) {
  // check inputs
  if(dstfile == NULL) {
    fprintf(stderr, "destination file not specified\n");
    return 0;
  }
  if((srcfile == NULL) && (qt == NULL) && (quality < 0)) {
    fprintf(stderr, "you must specify either srcfile, qt or quality\n");
    return 0;
  }
  if((srcfile == NULL) && (dct == NULL)) {
    fprintf(stderr, "you must specify either srcfile or dct\n");
    return 0;
  }

  // allocate
  struct jpeg_compress_struct cinfo_out;
  struct jpeg_error_mgr jerr_out;
  //memset((void *)&cinfo_out, 0x0, sizeof(struct jpeg_compress_struct));
  //memset((void *)&jerr_out, 0x0, sizeof(struct jpeg_error_mgr));

  // open the destination file
  FILE * fp_out;
  if ((fp_out = fopen(dstfile, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", dstfile);
    return 0;
  }

  // allocate
  struct jpeg_decompress_struct cinfo_in;
  struct jpeg_error_mgr jerr_in;
  FILE * fp_in;
  // read source jpeg
  if(srcfile != NULL)
    if((fp_in = _read_jpeg(srcfile, &cinfo_in, &jerr_in)) == NULL) return 0;

  cinfo_out.err = jpeg_std_error(&jerr_out);
  jpeg_create_compress(&cinfo_out);
  jpeg_stdio_dest(&cinfo_out, fp_out);

  if(srcfile != NULL) // copy critical parameters to dstfile
    jpeg_copy_critical_parameters((j_decompress_ptr)&cinfo_in,&cinfo_out);

  cinfo_out.image_width = image_dims[0];
  cinfo_out.image_height = image_dims[1];
  cinfo_out.in_color_space = in_color_space;
  if(in_components >= 0) cinfo_out.input_components = in_components;
  cinfo_out.num_components = cinfo_out.input_components;

  if(srcfile == NULL) // set defaults
    jpeg_set_defaults(&cinfo_out);

  // set sampling factors
  int chroma_factor[2];
  if(samp_factor != NULL) {
    chroma_factor[0] = *(samp_factor + 0);
    chroma_factor[1] = *(samp_factor + 1);
    for(int comp = 0; comp < cinfo_out.num_components; comp++) {
      cinfo_out.comp_info[comp].h_samp_factor = *(samp_factor + comp*2 + 0);
      cinfo_out.comp_info[comp].v_samp_factor = *(samp_factor + comp*2 + 1);
    }
  } else {
    chroma_factor[0] = cinfo_out.comp_info[0].h_samp_factor;
    chroma_factor[1] = cinfo_out.comp_info[0].v_samp_factor;
  }
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
  //fprintf(stderr, "chroma factors %d %d\n", chroma_factor[0], chroma_factor[1]);
  //for(int ch = 0; ch < 3; ch++) {
  //  fprintf(stderr, "sampling factors(%d) %d %d\n", ch, cinfo_out.comp_info[ch].v_samp_factor, cinfo_out.comp_info[ch].h_samp_factor);
  //}
  
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
  if(qt != NULL)
    for(int ch = 0; ch < 2; ch++)
      for(int i = 0; i < 64; i++)
        cinfo_out.quant_tbl_ptrs[ch]->quantval[i]/*(i&7)*8+(i>>3)*/ = qt[ch*64 + i];
        //qt[ch*64 + i] = cinfo_out.quant_tbl_ptrs[ch]->quantval[(i&7)*8+(i>>3)];
      //memcpy((void *)cinfo_out.quant_tbl_ptrs[ch]->quantval, (void *)(qt + ch*64), sizeof(short)*64);
  // write quality
  else if(quality > 0)
    jpeg_set_quality(&cinfo_out, quality, TRUE);

  //fprintf(stderr, "After setting of qt.\n");

  // DCT coefficients
  jvirt_barray_ptr *coeffs_array;
  if(srcfile != NULL) { // copy from source
    //fprintf(stderr, "Copy from source.\n");
    coeffs_array = jpeg_read_coefficients(&cinfo_in);
  } else { // allocate new
    //fprintf(stderr, "Create new array.\n");
    coeffs_array = (jvirt_barray_ptr *)(cinfo_out.mem->alloc_small)(
      (j_common_ptr)&cinfo_out,
      JPOOL_IMAGE,
      sizeof(jvirt_barray_ptr) * cinfo_out.num_components
    );
    //fprintf(stderr, "Set the coefficients. Image: %dx%d\n", cinfo_out.image_width, cinfo_out.image_height);
    for(int ch = 0; ch < (cinfo_out.num_components); ch++) { // channel iterator
      jpeg_component_info* comp_ptr = cinfo_out.comp_info + ch;
      //long v_samp_factor = *(samp_factor + ch*2 + 0);
      //long h_samp_factor = *(samp_factor + ch*2 + 1);
      comp_ptr->width_in_blocks = (JDIMENSION)ceil(((double)cinfo_out.image_width) / 8);// * comp_ptr->h_samp_factor / 4;
      comp_ptr->height_in_blocks = (JDIMENSION)ceil(((double)cinfo_out.image_height) / 8);// * comp_ptr->v_samp_factor / 4;
      if(ch > 0) {
        comp_ptr->width_in_blocks = ceil(((double)comp_ptr->width_in_blocks) / chroma_factor[0]);
        comp_ptr->height_in_blocks = ceil(((double)comp_ptr->height_in_blocks) / chroma_factor[1]);
      }
      coeffs_array[ch] = (cinfo_out.mem->request_virt_barray)(
        (j_common_ptr)&cinfo_out,
        JPOOL_IMAGE,
        TRUE,
        (JDIMENSION)jround_up(comp_ptr->width_in_blocks,  //component size in dct blocks (ignoring mcu)
		                          comp_ptr->h_samp_factor),   //round up is important, if border MCUs are not completely needed
        (JDIMENSION)jround_up(comp_ptr->height_in_blocks,
		                          comp_ptr->v_samp_factor),
		    (JDIMENSION)comp_ptr->v_samp_factor
      );

    }
    //fprintf(stderr, "ready to write coefficient\n");
  }
  #if JPEG_LIB_VERSION >= 80
  jpeg_calc_jpeg_dimensions(&cinfo_out);
  #endif
  //fprintf(stderr, "before writing coefficients\n");
  jpeg_write_coefficients(&cinfo_out,coeffs_array);
  //fprintf(stderr, "written coefficients\n");
  // write DCT coefficients
  if(dct != NULL) {
    JBLOCKARRAY buffer_one;
    JCOEFPTR blockptr_one;
    int HblocksY = cinfo_out.comp_info->height_in_blocks; // max height
    int WblocksY = cinfo_out.comp_info->width_in_blocks; // max width
    for(int ch = 0; ch < 3; ch++) { // channel iterator
      jpeg_component_info* comp_ptr = cinfo_out.comp_info + ch;
      int Hblocks = comp_ptr->height_in_blocks; // max height
      int Wblocks = comp_ptr->width_in_blocks; // max width
      for(int h = 0; h < Hblocks; h++) { // height iterator
        //fprintf(stderr, "accessing ch%d h%d/[H%d,W%d]\n", ch, h, Hblocks, Wblocks);
        buffer_one = (cinfo_out.mem->access_virt_barray)((j_common_ptr)&cinfo_out, coeffs_array[ch], h, (JDIMENSION)1, TRUE);
        for(int w = 0; w < Wblocks; w++) { // width iterator
          blockptr_one = buffer_one[0][w];
          for(int bh = 0; bh < 8; bh++)
            for(int bw = 0; bw < 8; bw++)
              blockptr_one[bh*8 + bw] = ((short *)_dct_offset(dct, ch, w, h, WblocksY, HblocksY))[bw*8 + bh];
        }
      }
    }
  }
  // cleanup
  jpeg_finish_compress( &cinfo_out );
  jpeg_destroy_compress( &cinfo_out );
  fclose( fp_out );

  if(srcfile != NULL) {
    jpeg_finish_decompress( &cinfo_in );
    jpeg_destroy_decompress( &cinfo_in );
    fclose( fp_in );
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
  unsigned long flags
) {

  // allocate
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  // read jpeg header
  FILE *fp;
  if((fp = _read_jpeg(srcfile, &cinfo, &jerr)) == NULL) return 0;

  // set parameters
  if(out_color_space >= 0) cinfo.out_color_space = out_color_space;
  else cinfo.out_color_space = cinfo.jpeg_color_space;
  //fprintf(stderr, "cjpeglib.c: out_color_space: %d %d\n", out_color_space, cinfo.out_color_space);
  
  if(dither_mode >= 0) cinfo.dither_mode = dither_mode;
  if(dct_method >= 0) cinfo.dct_method = dct_method;
  
  cinfo.do_fancy_upsampling   = 0 != (flags & DO_FANCY_UPSAMPLING);
  cinfo.do_block_smoothing    = 0 != (flags & DO_BLOCK_SMOOTHING);
  cinfo.quantize_colors       = 0 != (flags & QUANTIZE_COLORS);
  
  unsigned char *cmap[256];
  if(in_colormap != NULL)
    for(int i = 0; i < 256; i++) {
      cmap[i] = in_colormap + i*3;
      //if(i < 3) fprintf(stderr, " %d %d |", i, cmap[i][0]);
      //if(i == 255) fprintf(stderr, "\n");
    }
  if(flags & QUANTIZE_COLORS) {
    cinfo.actual_number_of_colors = 256; // TODO: parametrized
    cinfo.desired_number_of_colors = 256;
    if(in_colormap != NULL) cinfo.colormap = (char**)cmap;
  }
  
  cinfo.progressive_mode      = 0 != (flags & PROGRESSIVE_MODE);
  cinfo.arith_code            = 0 != (flags & ARITH_CODE);
  cinfo.CCIR601_sampling      = 0 != (flags & CCIR601_SAMPLING);
  cinfo.two_pass_quantize     = 0 != (flags & TWO_PASS_QUANTIZE);
  cinfo.enable_1pass_quant    = 0 != (flags & ENABLE_1PASS_QUANT);
  cinfo.enable_external_quant = 0 != (flags & ENABLE_EXTERNAL_QUANT);
  cinfo.enable_2pass_quant    = 0 != (flags & ENABLE_2PASS_QUANT);
  
  // decompress
  (void)jpeg_start_decompress(&cinfo);
  // read pixels
  unsigned char *rowptr = rgb;
  unsigned short stride = (flags & QUANTIZE_COLORS)?1:cinfo.out_color_components;
  while(cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, &rowptr, 1);
    rowptr += cinfo.output_width * stride;
  }
  // read quantization colormap
  if(flags & QUANTIZE_COLORS) {
    int N = cinfo.out_color_components;
    for(int ch=0; ch < N; ch++) {
      for(int i=0; i < 256; i++) {
        colormap[ch*256 + i] = cinfo.colormap[ch][i];
        //colormap[ch*256 + i] = cinfo.colormap[i][ch];
      }
    }
  }

  // cleanup
  (void) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(fp);

  return 1;
}

int write_jpeg_spatial(
  const char *srcfile,
  const char *dstfile,
  unsigned char *rgb,
  int *image_dims,
  int in_color_space,
  int in_components,
  int dct_method,
  int *samp_factor,
  unsigned short *qt,
  short quality,
  short smoothing_factor,
  unsigned long flags
) {

  // allocate
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  
  // open the destination file
  FILE * fp;
  if ((fp = fopen(dstfile, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", dstfile);
    return 0;
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);

  // set basic parameters
  if(srcfile == NULL) {
    cinfo.image_width = image_dims[0];
    cinfo.image_height = image_dims[1];
    if(in_color_space >= 0)
      cinfo.in_color_space = in_color_space;
    //fprintf(stderr, "writing 1 from cs %d to %d\n", cinfo.in_color_space, cinfo.jpeg_color_space);
    if(in_components >= 0)
      cinfo.input_components = in_components;
    cinfo.num_components = cinfo.input_components;
    jpeg_set_defaults(&cinfo);
  }

  // set advanced parameters
  if(dct_method >= 0) cinfo.dct_method = dct_method;
  int chroma_factor[2];
  if(samp_factor != NULL) {
    chroma_factor[0] = *(samp_factor + 0);
    chroma_factor[1] = *(samp_factor + 1);
    for(int comp = 0; comp < cinfo.num_components; comp++) {
      cinfo.comp_info[comp].h_samp_factor = *(samp_factor + comp*2 + 0);
      cinfo.comp_info[comp].v_samp_factor = *(samp_factor + comp*2 + 1);
    }
  } else {
    chroma_factor[0] = cinfo.comp_info[0].h_samp_factor;
    chroma_factor[1] = cinfo.comp_info[0].v_samp_factor;
  }

  // copy parameters
  struct jpeg_decompress_struct cinfo_in;
  struct jpeg_error_mgr jerr_in;
  FILE * fp_in = NULL;
  if(srcfile != NULL) {
    // read file
    if((fp_in = _read_jpeg(srcfile, &cinfo_in, &jerr_in)) == NULL) return 0;
    // decompress
    (void)jpeg_start_decompress(&cinfo_in);
    jpeg_copy_critical_parameters((j_decompress_ptr)&cinfo_in, (j_compress_ptr)&cinfo);
  }

  if(qt != NULL) {
    unsigned qt_u[64];
    // component 0
    for(int i = 0; i < 64; i++) qt_u[i] = qt[i]; // (i&7)*8+(i>>3)
    jpeg_add_quant_table(&cinfo, 0, qt_u, 100, FALSE);
    cinfo.comp_info[0].component_id = 0;
    cinfo.comp_info[0].quant_tbl_no = 0;
    // component 1
    for(int i = 0; i < 64; i++) qt_u[i] = qt[64 + i]; // (i&7)*8+(i>>3)
    jpeg_add_quant_table(&cinfo, 1, qt_u, 100, FALSE);
    cinfo.comp_info[1].component_id = 1;
    cinfo.comp_info[1].quant_tbl_no = 1;
    // component 2
    cinfo.comp_info[2].component_id = 2;
    cinfo.comp_info[2].quant_tbl_no = 1;
  } else if(quality >= 0) {
    jpeg_set_quality(&cinfo, quality, FALSE);
  }
  if(smoothing_factor >= 0) cinfo.smoothing_factor = smoothing_factor;
  if(in_color_space >= 0) {
    cinfo.in_color_space = in_color_space;
    //jpeg_set_colorspace(&cinfo, in_color_space);
    //fprintf(stderr, "writing 2 from cs %d to %d\n", cinfo.in_color_space, cinfo.jpeg_color_space);
  }
  //fprintf(stderr, "colorspace conversion %d -> %d\n", cinfo.in_color_space, cinfo.jpeg_color_space);
  
  cinfo.progressive_mode   = 0 != (flags & PROGRESSIVE_MODE);
  cinfo.optimize_coding    = 0 != (flags & OPTIMIZE_CODING);
  cinfo.arith_code         = 0 != (flags & ARITH_CODE);
  cinfo.write_JFIF_header  = 0 != (flags & WRITE_JFIF_HEADER);
  cinfo.write_Adobe_marker = 0 != (flags & WRITE_ADOBE_MARKER);
  cinfo.CCIR601_sampling   = 0 != (flags & CCIR601_SAMPLING);

  // https://gist.github.com/kentakuramochi/f64e7646f1db8335c80f131be8359044

  // write data
  unsigned char *rowptr = rgb;
  jpeg_start_compress(&cinfo, TRUE);
  for(unsigned h = 0; h < cinfo.image_height; h++) {
    jpeg_write_scanlines(&cinfo, &rowptr, 1);
    rowptr += cinfo.image_width * cinfo.input_components;
  }

  // cleanup
  jpeg_finish_compress( &cinfo );
  jpeg_destroy_compress( &cinfo );
  fclose( fp );
  if(srcfile != NULL) {
    jpeg_destroy_decompress( &cinfo_in );
    fclose( fp_in );
  }
  
  return 1;
}


int print_jpeg_params(const char *srcfile)
{
  // allocate
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  // read jpeg header
  FILE *fp;
  if((fp = _read_jpeg(srcfile, &cinfo, &jerr)) == NULL) return 0;

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

  for(int comp = 0; comp < cinfo.out_color_components; comp++) {
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



#ifdef __cplusplus
}
#endif