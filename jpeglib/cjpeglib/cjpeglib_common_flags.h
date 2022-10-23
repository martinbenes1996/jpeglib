#ifndef CJPEGLIB_COMMON_FLAGS_HPP
#define CJPEGLIB_COMMON_FLAGS_HPP

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long BITMASK;

#ifdef __cplusplus
#define FLAG_SET boolean
#else
#include <stdbool.h>
#define FLAG_SET bool
#endif

FLAG_SET flag_is_set(
	BITMASK flags,
	BITMASK mask
);
unsigned char overwrite_flag(
	BITMASK flags,
	BITMASK mask
);

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

#ifdef __cplusplus
}
#endif

#endif // CJPEGLIB_COMMON_FLAGS_HPP