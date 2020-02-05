#ifndef NNGPP_PAIR1_H
#define NNGPP_PAIR1_H
#include <nngpp/socket.h>
#include <nng/protocol/pair1/pair.h>

namespace nng { namespace pair {
inline namespace v1 {

inline socket open() {
	nng_socket s;
	int r = nng_pair1_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pair1_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_pair1_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pair1_open_raw");
	}
	return socket(s);
}

enum class option {
	poly // bool
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_PAIR1_POLY
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline bool get_opt_poly( socket_view s ) {
	return s.get_opt_bool( to_name(option::poly) );
}

inline void set_opt_poly( socket_view s, bool v ) {
	s.set_opt_bool( to_name(option::poly), v );
}

}
}}

#endif
