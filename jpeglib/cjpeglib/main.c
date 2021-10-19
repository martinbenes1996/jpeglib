
#include <stdio.h>
#include <stdlib.h>
#include "cstegojpeg.h"

int main() {
    // load info
    const char *srcfile = "../../img/IMG_0791.jpeg";
    int sizes[6] = {0,0,0,0,0,0};
    int n_comps;
    if(!read_jpeg_info(srcfile, sizes, &n_comps)) exit(1);
    //printf("Image: %d channels of %dx%d blocks (8x8)\n", n_comps, sizes[0], sizes[1]);
    
    // load dct
    short *dct_buffer = (short *)malloc(sizeof(short)*sizes[0]*sizes[1]*8*8*n_comps);
    short *qt_buffer = (short *)malloc(sizeof(short)*8*8*n_comps);
    read_jpeg_data(srcfile, dct_buffer, qt_buffer);

    // steganography over dct_buffer
    // todo

    // write dct
    const char * dstfile = "../../c_output.jpeg";
    write_jpeg_data(srcfile, dstfile, dct_buffer);

    // cleanup
    free(dct_buffer);
    free(qt_buffer);

    return 1;
}