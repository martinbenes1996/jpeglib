
// ---------- Meta -------------
int read_jpeg_info(
    const char *srcfile,
    int *dct_dims,
    int *image_dims,
    int *num_components,
    int *samp_factor,
    int *jpeg_color_space
);

// ----------- DCT -------------
int read_jpeg_dct(
    const char *srcfile,
    short *dct,
    unsigned short *qt
);
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
);

// ----------- RGB -------------
int read_jpeg_spatial(
    const char *srcfile,
    unsigned char *rgb,
    unsigned char *colormap, // colormap used
    unsigned char *in_colormap, // colormap to use
    int out_color_space,
    int dither_mode,
    int dct_method,
    unsigned long flags
);

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
);

//int jpeg_lib_version(void) { return JPEG_LIB_VERSION; }
int print_jpeg_params(const char *srcfile);
