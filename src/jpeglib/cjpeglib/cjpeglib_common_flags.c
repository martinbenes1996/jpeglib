
#ifdef __cplusplus
extern "C" {
#endif

#include "cjpeglib_common_flags.h"


boolean flag_to_boolean_value(
	BITMASK flags,
	BITMASK mask
) {
	// Flags are encoded as 2 bits. The less significant bit is the new value to be set.
	// This method returns *false* when the LSB is set to 0, and *true* when the LSB is set to 1.
	return (flags & mask) != 0;
}
boolean overwrite_default(
	BITMASK flags,
	BITMASK mask
) {
	// Flags are encoded as 2 bits. The more significant bit indicates whether the default should be kept (= 1) or overwritten (= 0).
	// This method returns 1 when the default should be overwritten. It returns 0 when the default should be kept.
	return ((flags & mask) != 0);
}

#ifdef __cplusplus
}
#endif