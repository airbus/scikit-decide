#ifndef NNGPP_PLATFORM_H
#define NNGPP_PLATFORM_H
#include "cv.h"
#include "mtx.h"
#include "thread.h"

namespace nng {

inline nng_time clock() noexcept {
	return nng_clock();
}

inline void msleep( nng_duration dt ) noexcept {
	nng_msleep(dt);
}

inline uint32_t random() noexcept {
	return nng_random();
}

}

#endif
