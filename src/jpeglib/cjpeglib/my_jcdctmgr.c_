
#define PROPAGATE_PUBLIC_API
#define JPEG_INTERNALS
#include "vjpeglib.h"

METHODDEF(void)
my_forward_DCT(
    j_compress_ptr cinfo,
    jpeg_component_info * compptr,
	JSAMPARRAY sample_data,
    JBLOCKROW coef_blocks,
	JDIMENSION start_row,
    JDIMENSION start_col,
	JDIMENSION num_blocks)
{
    /* This routine is heavily used, so it's worth coding it tightly. */
    my_fdct_ptr fdct = (my_fdct_ptr) cinfo->fdct;
    #if JPEG_LIB_VERSION < 70
        forward_DCT_method_ptr do_dct = fdct->do_dct;
    #else
        forward_DCT_method_ptr do_dct = fdct->do_dct[compptr->component_index];
    #endif
    DCTELEM * divisors = fdct->divisors[compptr->quant_tbl_no];
    DCTELEM workspace[DCTSIZE2];	/* work area for FDCT subroutine */
    JDIMENSION bi;
    #if JPEG_LIB_VERSION < 70
        JDIMENSION maxbi = DCTSIZE;
    #else
        JDIMENSION maxbi = compptr->DCT_h_scaled_size;
    #endif

    #if JPEG_LIB_VERSION >= 70
        sample_data += start_row;	/* fold in the vertical offset once */
    #endif

    for (bi = 0; bi < num_blocks; bi++, start_col += maxbi) {

        #if JPEG_LIB_VERSION < 70
        /* Load data into workspace, applying unsigned->signed conversion */
        {
            register DCTELEM *workspaceptr;
            register JSAMPROW elemptr;
            register int elemr;

            workspaceptr = workspace;
            for (elemr = 0; elemr < DCTSIZE; elemr++) {
                elemptr = sample_data[elemr] + start_col;
                #if DCTSIZE == 8		/* unroll the inner loop */
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                #else
                    {
                    register int elemc;
                    for (elemc = DCTSIZE; elemc > 0; elemc--) {
                        *workspaceptr++ = GETJSAMPLE(*elemptr++) - CENTERJSAMPLE;
                    }
                    }
                #endif
            }
        }
        #endif

        /* Perform the DCT */
        #if JPEG_LIB_VERSION >= 70
            (*do_dct) (workspace, sample_data, start_col);
        #else
            (*do_dct) (workspace);
        #endif

        /* No quantization - store unquantized inside JPEG */
        /* Quantize/descale the coefficients, and store into coef_blocks[] */
        {
            register DCTELEM temp, qval;
            register int i;
            register JCOEFPTR output_ptr = coef_blocks[bi];

            for (i = 0; i < DCTSIZE2; i++) {
                qval = divisors[i];
                temp = workspace[i];
                // fprintf(stderr, "%d/%x ", temp, qval);
                output_ptr[i] = (JCOEF) temp;
            }
            // fprintf(stderr, "\n");
        }

    }
}

#ifdef DCT_FLOAT_SUPPORTED

METHODDEF(void)
my_forward_DCT_float(
    j_compress_ptr cinfo,
    jpeg_component_info * compptr,
	JSAMPARRAY sample_data,
    JBLOCKROW coef_blocks,
	JDIMENSION start_row,
    JDIMENSION start_col,
	JDIMENSION num_blocks)
/* This version is used for floating-point DCT implementations. */
{
    /* This routine is heavily used, so it's worth coding it tightly. */
    my_fdct_ptr fdct = (my_fdct_ptr) cinfo->fdct;
    #if JPEG_LIB_VERSION < 70
        float_DCT_method_ptr do_float_dct = fdct->do_float_dct;
    #else
        float_DCT_method_ptr do_float_dct = fdct->do_float_dct[compptr->component_index];
    #endif
    // FAST_FLOAT * divisors = fdct->float_divisors[compptr->quant_tbl_no];
    FAST_FLOAT workspace[DCTSIZE2]; /* work area for FDCT subroutine */
    JDIMENSION bi;
    #if JPEG_LIB_VERSION < 70
        JDIMENSION maxbi = DCTSIZE;
    #else
        JDIMENSION maxbi = compptr->DCT_h_scaled_size;
    #endif

    #if JPEG_LIB_VERSION >= 70
        sample_data += start_row;	/* fold in the vertical offset once */
    #endif

    for (bi = 0; bi < num_blocks; bi++, start_col += maxbi) {

        #if JPEG_LIB_VERSION < 70
        /* Load data into workspace, applying unsigned->signed conversion */
        {
            register FAST_FLOAT *workspaceptr;
            register JSAMPROW elemptr;
            register int elemr;

            workspaceptr = workspace;
            for (elemr = 0; elemr < DCTSIZE; elemr++) {
                elemptr = sample_data[elemr] + start_col;
                #if DCTSIZE == 8		/* unroll the inner loop */
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    *workspaceptr++ = (FAST_FLOAT)(GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                #else
                {
                    register int elemc;
                    for (elemc = DCTSIZE; elemc > 0; elemc--) {
                        *workspaceptr++ = (FAST_FLOAT)
                        (GETJSAMPLE(*elemptr++) - CENTERJSAMPLE);
                    }
                }
                #endif
            }
        }
        #endif

        /* Perform the DCT */
        #if JPEG_LIB_VERSION >= 70
            (*do_float_dct) (workspace, sample_data, start_col);
        #else
            (*do_float_dct) (workspace);
        #endif

        /* Quantize/descale the coefficients, and store into coef_blocks[] */
        {
            register FAST_FLOAT temp;
            register int i;
            register JCOEFPTR output_ptr = coef_blocks[bi];

            for (i = 0; i < DCTSIZE2; i++) {
                temp = workspace[i];
                output_ptr[i] = (JCOEF) temp;
            }
        }
    }
}

#endif /* DCT_FLOAT_SUPPORTED */


void set_dct_callback(
	j_compress_ptr cinfo)
{
    my_fdct_ptr fdct = (my_fdct_ptr)cinfo->fdct;
    #if JPEG_LIB_VERSION < 70
        switch (cinfo->dct_method) {
            #ifdef DCT_ISLOW_SUPPORTED
            case JDCT_ISLOW:
                fdct->pub.forward_DCT = my_forward_DCT;
                break;
            #endif
            #ifdef DCT_IFAST_SUPPORTED
            case JDCT_IFAST:
                fdct->pub.forward_DCT = my_forward_DCT;
                break;
            #endif
            #ifdef DCT_FLOAT_SUPPORTED
            case JDCT_FLOAT:
                fdct->pub.forward_DCT = my_forward_DCT_float;
                break;
            #endif
            default:
                ERREXIT(cinfo, JERR_NOT_COMPILED);
                break;
        }

    #else // JPEG_LIB_VERSION >= 70
        for(int ci = 0; ci < cinfo->num_components; ci++) {
            switch (cinfo->dct_method) {
                #ifdef DCT_ISLOW_SUPPORTED
                case JDCT_ISLOW:
                    fdct->pub.forward_DCT[ci] = my_forward_DCT;
                    break;
                #endif
                #ifdef DCT_IFAST_SUPPORTED
                case JDCT_IFAST:
                    fdct->pub.forward_DCT[ci] = my_forward_DCT;
                    break;
                #endif
                #ifdef DCT_FLOAT_SUPPORTED
                case JDCT_FLOAT:
                    fdct->pub.forward_DCT[ci] = my_forward_DCT_float;
                    break;
                #endif
                default:
                    ERREXIT(cinfo, JERR_NOT_COMPILED);
                    break;
            }
        }
    #endif
}

