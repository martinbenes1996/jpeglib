#ifdef __cplusplus
extern "C" {
#endif

// https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Desktop-generic/LSB-Desktop-generic/libjpegman.html

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

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

//#include "cstegojpeg.h"

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
  if(jpeg_color_space != NULL)
    jpeg_color_space[0] = cinfo.jpeg_color_space;

  // cleanup
  jpeg_destroy_decompress( &cinfo );
  fclose(fp);

  return 1;
}

void *_dct_offset(short * base, int channel, int w, int h, int Wmax, int Hmax)
{
  return (void *)(base + 64*(h + Hmax*(w + Wmax*(channel))));
}

int read_jpeg_dct(const char *srcfile, short *dct, short *qt)
{ 
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
  
  // read quantization table
  if(qt != NULL) {
    j_decompress_ptr dec_cinfo  = (j_decompress_ptr) &cinfo;
    for(int ch = 0; ch < 2; ch++) {
      jpeg_component_info *ci_ptr = &dec_cinfo->comp_info[ch];
      JQUANT_TBL *tbl = ci_ptr->quant_table;
      memcpy((void *)(qt + ch*64), (void*)tbl->quantval, sizeof(short)*64);
    }
  }

  // cleanup
  jpeg_finish_decompress( &cinfo );
  jpeg_destroy_decompress( &cinfo );
  fclose(fp);

  return 1;
}

// I plan to remove srcfile argument and add qt, so that it creates brand new file
// without copying the old one
int write_jpeg_dct(const char *srcfile, const char *dstfile, short *dct)
{
  // allocate
  struct jpeg_decompress_struct cinfo_in;
  struct jpeg_error_mgr jerr_in;
  struct jpeg_compress_struct cinfo_out;
  struct jpeg_error_mgr jerr_out;
  
  // read jpeg header
  FILE * fp_in;
  if((fp_in = _read_jpeg(srcfile, &cinfo_in, &jerr_in)) == NULL) return 0;
  
  // read DCT coefficients
  jvirt_barray_ptr *coeffs_array = jpeg_read_coefficients(&cinfo_in);
  
  // open the destination file
  FILE * fp_out;
  if ((fp_out = fopen(dstfile, "wb")) == NULL) {
    fprintf(stderr, "can't open %s\n", dstfile);
    return 0;
  }

  cinfo_out.err = jpeg_std_error(&jerr_out);
  jpeg_create_compress(&cinfo_out);
  jpeg_stdio_dest(&cinfo_out, fp_out);

  j_compress_ptr cinfo_out_ptr = &cinfo_out;
  jpeg_copy_critical_parameters((j_decompress_ptr)&cinfo_in,cinfo_out_ptr);
  //cinfo_out.comp_info[0].h_samp_factor = 2;
  jpeg_write_coefficients(cinfo_out_ptr, coeffs_array);

  // read dct
  JBLOCKARRAY buffer_one;
  JCOEFPTR blockptr_one;
  int HblocksY = cinfo_out.comp_info->height_in_blocks; // max height
  int WblocksY = cinfo_out.comp_info->width_in_blocks; // max width
  for(int ch = 0; ch < 3; ch++) { // channel iterator
    jpeg_component_info* compptr_one = cinfo_out.comp_info + ch;
    int Hblocks = compptr_one->height_in_blocks; // max height
    int Wblocks = compptr_one->width_in_blocks; // max width
    for(int h = 0; h < Hblocks; h++) { // height iterator
      buffer_one = (cinfo_out.mem->access_virt_barray)((j_common_ptr)&cinfo_out, coeffs_array[ch], h, (JDIMENSION)1, FALSE);
      for(int w = 0; w < Wblocks; w++) { // width iterator
        blockptr_one = buffer_one[0][w];
        for(int bh = 0; bh < 8; bh++) {
          for(int bw = 0; bw < 8; bw++) {
            int i = bw*8 + bh;
            blockptr_one[bh*8 + bw] = ((short *)_dct_offset(dct, ch, w, h, WblocksY, HblocksY))[i];
          }
        }
        //memcpy((void *)blockptr_one, _dct_offset(dct, ch, w, h, WblocksY, HblocksY), sizeof(short)*64);
      }
    }
  }
  
  // cleanup
  jpeg_finish_compress( &cinfo_out );
  jpeg_destroy_compress( &cinfo_out );
  jpeg_finish_decompress( &cinfo_in );
  jpeg_destroy_decompress( &cinfo_in );
  fclose( fp_in );
  fclose( fp_out );
  return 1;
}

int read_jpeg_spatial(
  const char *srcfile,
  unsigned char *rgb,
  int out_color_space,
  int dither_mode,
  int dct_method,
  int *samp_factor,
  unsigned flags
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
  
  if(dither_mode >= 0) cinfo.dither_mode = dither_mode;
  if(dct_method >= 0) cinfo.dct_method = dct_method;
  
  cinfo.do_fancy_upsampling   = 0 != (flags & DO_FANCY_UPSAMPLING);
  cinfo.do_block_smoothing    = 0 != (flags & DO_BLOCK_SMOOTHING);
  cinfo.quantize_colors       = 0 != (flags & QUANTIZE_COLORS);
  cinfo.progressive_mode      = 0 != (flags & PROGRESSIVE_MODE);
  cinfo.arith_code            = 0 != (flags & ARITH_CODE);
  cinfo.CCIR601_sampling      = 0 != (flags & CCIR601_SAMPLING);
  cinfo.two_pass_quantize     = 0 != (flags & TWO_PASS_QUANTIZE);
  cinfo.enable_1pass_quant    = 0 != (flags & ENABLE_1PASS_QUANT);
  cinfo.enable_external_quant = 0 != (flags & ENABLE_EXTERNAL_QUANT);
  cinfo.enable_2pass_quant    = 0 != (flags & ENABLE_2PASS_QUANT);

  // decompress
  (void)jpeg_start_decompress(&cinfo);

  //fprintf(stderr, "  - Color spaces: %d %d\n", cinfo.out_color_space, cinfo.jpeg_color_space);
  //fprintf(stderr, "  - Decompression: %d %d\n", cinfo.dither_mode, cinfo.dct_method);
  //fprintf(stderr, "  - Flags: %u\n", flags);
  //fprintf(stderr, "  - Params: |%u %u| %u\n", cinfo.output_width, cinfo.output_height, cinfo.out_color_components);

  // read
  unsigned char *rowptr = rgb;
  while(cinfo.output_scanline < cinfo.output_height) {
    jpeg_read_scanlines(&cinfo, &rowptr, 1);
    rowptr += cinfo.output_width * cinfo.out_color_components;
  }

  if(samp_factor != NULL)
    for(int comp = 0; comp < cinfo.out_color_components; comp++) {
      *(samp_factor + comp*2 + 0) = cinfo.comp_info[comp].h_samp_factor;
      *(samp_factor + comp*2 + 1) = cinfo.comp_info[comp].v_samp_factor;
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
  short quality,
  unsigned *qt,
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
  cinfo.image_width = image_dims[0];
  cinfo.image_height = image_dims[1];
  cinfo.in_color_space = in_color_space;
  if(in_components >= 0) cinfo.input_components = in_components;
  cinfo.num_components = cinfo.input_components;
  jpeg_set_defaults(&cinfo);
  
  
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
  //fprintf(stderr, "After jpeg_copy_critical_parameters.\n");
  

  // set advanced parameters
  if(dct_method >= 0) cinfo.dct_method = dct_method;
  if(samp_factor != NULL)
    for(int comp = 0; comp < cinfo.input_components; comp++) {
      cinfo.comp_info[comp].h_samp_factor = *(samp_factor + comp*2 + 0);
      cinfo.comp_info[comp].v_samp_factor = *(samp_factor + comp*2 + 1);
    }

  if(qt != NULL) {
    jpeg_add_quant_table(&cinfo, 2, qt, 100, TRUE);
    cinfo.comp_info[0].quant_tbl_no = 2;
    jpeg_add_quant_table(&cinfo, 3, qt + 64, 100, TRUE);
    cinfo.comp_info[1].quant_tbl_no = 3;
    cinfo.comp_info[1].quant_tbl_no = 3;
  } else if(quality >= 0) jpeg_set_quality(&cinfo, quality, TRUE);
  if(smoothing_factor >= 0) cinfo.smoothing_factor = smoothing_factor;
  
  cinfo.progressive_mode   = 0 != (flags & PROGRESSIVE_MODE);
  cinfo.optimize_coding    = 0 != (flags & OPTIMIZE_CODING);
  cinfo.arith_code         = 0 != (flags & ARITH_CODE);
  cinfo.write_JFIF_header  = 0 != (flags & WRITE_JFIF_HEADER);
  cinfo.write_Adobe_marker = 0 != (flags & WRITE_ADOBE_MARKER);
  //cinfo.CCIR601_sampling   = 0 != (flags & CCIR601_SAMPLING);

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