
// ---------- Meta -------------
typedef unsigned long BITMASK;

int read_jpeg_info(
    const char *srcfile,
    int *block_dims,
    int *image_dims,
    int *num_components,
    int *samp_factor,
    int *jpeg_color_space,
    int *marker_lengths,
    int *mark_types,
    BITMASK *flags);

int read_jpeg_markers(
    const char *srcfile,
    unsigned char *markers);

// ----------- DCT -------------
int read_jpeg_dct(
    const char *srcfile,
    short *Y,
    short *Cb,
    short *Cr,
    unsigned short *qt,
    unsigned char *quant_tbl_no);
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
    unsigned char *markers);

// ----------- RGB -------------
int read_jpeg_spatial(
    const char *srcfile,
    unsigned char *rgb,
    unsigned char *colormap,    // colormap used
    unsigned char *in_colormap, // colormap to use
    int out_color_space,
    int dither_mode,
    int dct_method,
    BITMASK flags);

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
    BITMASK flags);

// int jpeg_lib_version(void) { return JPEG_LIB_VERSION; }
int print_jpeg_params(const char *srcfile);
