#ifndef NNGPP_REQ0_H
#define NNGPP_REQ0_H
#include <nngpp/ctx.h>
#include <nng/protocol/reqrep0/req.h>

namespace nng { namespace req {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_req0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_req0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_req0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_req0_open_raw");
	}
	return socket(s);
}

enum class option {
	resend_time // ms
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_REQ_RESENDTIME
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline nng_duration get_opt_resend_time( socket_view s ) {
	return s.get_opt_ms( to_name(option::resend_time) );
}

inline void set_opt_resend_time( socket_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::resend_time), v );
}

inline nng_duration get_opt_resend_time( ctx_view s ) {
	return s.get_opt_ms( to_name(option::resend_time) );
}

inline void set_opt_resend_time( ctx_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::resend_time), v );
}

}
}}

#endif
