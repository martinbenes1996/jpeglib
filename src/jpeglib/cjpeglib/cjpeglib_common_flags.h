#ifndef CJPEGLIB_COMMON_FLAGS_HPP
#define CJPEGLIB_COMMON_FLAGS_HPP

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include "vjpeglib.h"

typedef unsigned int BITMASK;

boolean flag_to_boolean_value(
	BITMASK flags,
	BITMASK mask
);

boolean overwrite_default(
	BITMASK flags,
	BITMASK mask
);

// There are 2 bits per flag. The more significant bit (MSB) indicates whether to keep the default value (= 1) or whether to overwrite the default value (= 0). If the MSB is 1, the less significant bit (LSB) is ignored. If the MSB is 0, the LSB is taken as the new value.
// #define DO_FANCY_UPSAMPLING ((BITMASK)0b1 << 0)
// #define DO_BLOCK_SMOOTHING ((BITMASK)0b1 << 2)
// #define TWO_PASS_QUANTIZE ((BITMASK)0b1 << 4)
// #define ENABLE_1PASS_QUANT ((BITMASK)0b1 << 6)
// #define ENABLE_EXTERNAL_QUANT ((BITMASK)0b1 << 8)
// #define ENABLE_2PASS_QUANT ((BITMASK)0b1 << 10)
// #define OPTIMIZE_CODING ((BITMASK)0b1 << 12)
// #define PROGRESSIVE_MODE ((BITMASK)0b1 << 14)
// #define QUANTIZE_COLORS ((BITMASK)0b1 << 16)
// #define ARITH_CODE ((BITMASK)0b1 << 18)
// #define WRITE_JFIF_HEADER ((BITMASK)0b1 << 20)
// #define WRITE_ADOBE_MARKER ((BITMASK)0b1 << 22)
// #define CCIR601_SAMPLING ((BITMASK)0b1 << 24)
// #define FORCE_BASELINE ((BITMASK)0b1 << 26)
// #define TRELLIS_QUANT ((BITMASK)0b1 << 28)
// #define TRELLIS_QUANT_DC ((BITMASK)0b1 << 30)
// #define TRELLIS_Q_OPT ((BITMASK)0b1 << 32)
// #define OPTIMIZE_SCANS ((BITMASK)0b1 << 34)
// #define USE_SCANS_IN_TRELLIS ((BITMASK)0b1 << 36)
// #define OVERSHOOT_DERINGING ((BITMASK)0b1 << 38)

#define DO_FANCY_UPSAMPLING ((BITMASK)0b1 << 0)
#define DO_BLOCK_SMOOTHING ((BITMASK)0b1 << 1)
#define TWO_PASS_QUANTIZE ((BITMASK)0b1 << 2)
#define ENABLE_1PASS_QUANT ((BITMASK)0b1 << 3)
#define ENABLE_EXTERNAL_QUANT ((BITMASK)0b1 << 4)
#define ENABLE_2PASS_QUANT ((BITMASK)0b1 << 5)
#define OPTIMIZE_CODING ((BITMASK)0b1 << 6)
#define PROGRESSIVE_MODE ((BITMASK)0b1 << 7)
#define QUANTIZE_COLORS ((BITMASK)0b1 << 8)
#define ARITH_CODE ((BITMASK)0b1 << 9)
#define WRITE_JFIF_HEADER ((BITMASK)0b1 << 10)
#define WRITE_ADOBE_MARKER ((BITMASK)0b1 << 11)
#define CCIR601_SAMPLING ((BITMASK)0b1 << 12)
#define FORCE_BASELINE ((BITMASK)0b1 << 13)
#define TRELLIS_QUANT ((BITMASK)0b1 << 14)
#define TRELLIS_QUANT_DC ((BITMASK)0b1 << 15)
#define TRELLIS_Q_OPT ((BITMASK)0b1 << 16)
#define OPTIMIZE_SCANS ((BITMASK)0b1 << 17)
#define USE_SCANS_IN_TRELLIS ((BITMASK)0b1 << 18)
#define OVERSHOOT_DERINGING ((BITMASK)0b1 << 19)

#ifdef __cplusplus
}
#endif

#endif // CJPEGLIB_COMMON_FLAGS_HPP