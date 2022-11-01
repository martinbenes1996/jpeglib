
#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib_common_flags.h"

boolean flag_is_set(
	BITMASK flags,
	BITMASK mask
) {
	return (flags & mask) != 0;
}
unsigned char overwrite_flag(
	BITMASK flags,
	BITMASK mask
) {
	return ((flags & (mask << 1)) == 0);
}

#ifdef __cplusplus
}
#endif