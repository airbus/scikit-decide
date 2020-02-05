#ifndef NNGPP_ERROR_H
#define NNGPP_ERROR_H
#include <nng/nng.h>
#include <exception>

namespace nng {

enum class error {
	success     = 0,
	intr        = NNG_EINTR,
	nomem       = NNG_ENOMEM,
	inval       = NNG_EINVAL,
	busy        = NNG_EBUSY,
	timedout    = NNG_ETIMEDOUT,
	connrefused = NNG_ECONNREFUSED,
	closed      = NNG_ECLOSED,
	again       = NNG_EAGAIN,
	notsup      = NNG_ENOTSUP,
	addrinuse   = NNG_EADDRINUSE,
	state       = NNG_ESTATE,
	noent       = NNG_ENOENT,
	proto       = NNG_EPROTO,
	unreachable = NNG_EUNREACHABLE,
	addrinval   = NNG_EADDRINVAL,
	perm        = NNG_EPERM,
	msgsize     = NNG_EMSGSIZE,
	connaborted = NNG_ECONNABORTED,
	connreset   = NNG_ECONNRESET,
	connshut    = NNG_ECONNSHUT,
	canceled    = NNG_ECANCELED,
	nofiles     = NNG_ENOFILES,
	nospc       = NNG_ENOSPC,
	exist       = NNG_EEXIST,
	readonly    = NNG_EREADONLY,
	writeonly   = NNG_EWRITEONLY,
	crypto      = NNG_ECRYPTO,
	peerauth    = NNG_EPEERAUTH,
	noarg       = NNG_ENOARG,
	ambiguous   = NNG_EAMBIGUOUS,
	badtype     = NNG_EBADTYPE,
	internal    = NNG_EINTERNAL,
	syserr      = NNG_ESYSERR,
	tranerr     = NNG_ETRANERR
};

inline const char* to_string( error e ) noexcept {
	return nng_strerror( (int)e );
}

class exception : public std::exception {
	const char* source;
	int err;
public:
	
	explicit exception( int e, const char* s = "" ) noexcept : source(s), err(e) {}

	explicit exception( error e, const char* s = "" ) noexcept : exception((int)e,s) {}
	
	error get_error() const noexcept {
		return (error)err;
	}

	const char* who() const noexcept {
		return source;
	}
	
	const char* what() const noexcept final {
		return nng_strerror(err);
	}
};

}

#endif
