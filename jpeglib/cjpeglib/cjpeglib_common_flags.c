
#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib_common_flags.h"

FLAG_SET flag_is_set(
	BITMASK flags,
	BITMASK mask
) {
	return (flags & mask) != 0;
}
FLAG_SET overwrite_flag(
	BITMASK flags,
	BITMASK mask
) {
	return (flags & (mask << 1)) == 0;
}

#ifdef __cplusplus
}
#endif