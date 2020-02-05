#ifndef NNGPP_SUB0_H
#define NNGPP_SUB0_H
#include <nngpp/socket.h>
#include <nng/protocol/pubsub0/sub.h>

namespace nng { namespace sub {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_sub0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_sub0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_sub0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_sub0_open_raw");
	}
	return socket(s);
}

enum class option {
	subscribe,   // opaque
	unsubscribe, // opaque
	prefnew      // bool
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_SUB_SUBSCRIBE,
	NNG_OPT_SUB_UNSUBSCRIBE,
	NNG_OPT_SUB_PREFNEW
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline void set_opt_subscribe( socket_view s, view v ) {
	s.set_opt( to_name(option::subscribe), v );
}

inline void set_opt_unsubscribe( socket_view s, view v ) {
	s.set_opt( to_name(option::unsubscribe), v );
}

inline bool get_opt_prefnew( socket_view s ) {
	return s.get_opt_bool( to_name(option::prefnew) );
}

inline void set_opt_prefnew( socket_view s, bool v ) {
	s.set_opt_bool( to_name(option::prefnew), v );
}

}
}}

#endif
