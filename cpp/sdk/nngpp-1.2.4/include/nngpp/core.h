#ifndef NNGPP_CORE_H
#define NNGPP_CORE_H
#include <nng/nng.h>

namespace nng {

enum class sockaddr_family {
	unspec = NNG_AF_UNSPEC,
	inproc = NNG_AF_INPROC,
	ipc    = NNG_AF_IPC,
	inet   = NNG_AF_INET,
	inet6  = NNG_AF_INET6,
	zt     = NNG_AF_ZT
};

enum class pipe_ev {
	add_pre  = NNG_PIPE_EV_ADD_PRE,
	add_post = NNG_PIPE_EV_ADD_POST,
	rem_post = NNG_PIPE_EV_REM_POST,
	num      = NNG_PIPE_EV_NUM
};

namespace flag {
enum {
	alloc    = NNG_FLAG_ALLOC,
	nonblock = NNG_FLAG_NONBLOCK
};
}

inline const char* version() noexcept {
	return nng_version();
}

inline void fini() noexcept {
	nng_fini();
}

inline void close_all() noexcept {
	nng_closeall();
}

}

#endif
