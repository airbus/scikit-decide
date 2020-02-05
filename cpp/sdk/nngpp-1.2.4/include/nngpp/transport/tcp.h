#ifndef NNGPP_TRANSPORT_TCP_H
#define NNGPP_TRANSPORT_TCP_H
#include <nngpp/listener_view.h>
#include <nngpp/dialer_view.h>
#include <nngpp/pipe_view.h>
#include <nngpp/socket_view.h>
#include <nng/transport/tcp/tcp.h>

namespace nng { namespace tcp {

inline void register_transport() {
	int r = nng_tcp_register();
	if( r != 0 ) {
		throw exception(r,"nng_tcp_register");
	}
}

enum class option {
	no_delay,
	keep_alive,
	bound_port
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_TCP_NODELAY,   // bool
	NNG_OPT_TCP_KEEPALIVE, // bool
	NNG_OPT_TCP_BOUND_PORT // int
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline bool get_opt_no_delay( nng::pipe_view s ) {
	return s.get_opt_bool( to_name(option::no_delay) );
}

inline bool get_opt_keep_alive( nng::pipe_view s ) {
	return s.get_opt_bool( to_name(option::keep_alive) );
}

inline int get_opt_bound_port( nng::pipe_view s ) {
	return s.get_opt_int( to_name(option::bound_port) );
}

inline bool get_opt_no_delay( nng::dialer_view s ) {
	return s.get_opt_bool( to_name(option::no_delay) );
}

inline bool get_opt_keep_alive( nng::dialer_view s ) {
	return s.get_opt_bool( to_name(option::keep_alive) );
}

inline bool get_opt_no_delay( nng::listener_view s ) {
	return s.get_opt_bool( to_name(option::no_delay) );
}

inline bool get_opt_keep_alive( nng::listener_view s ) {
	return s.get_opt_bool( to_name(option::keep_alive) );
}

inline int get_opt_bound_port( nng::listener_view s ) {
	return s.get_opt_int( to_name(option::bound_port) );
}

inline bool get_opt_no_delay( nng::socket_view s ) {
	return s.get_opt_bool( to_name(option::no_delay) );
}

inline bool get_opt_keep_alive( nng::socket_view s ) {
	return s.get_opt_bool( to_name(option::keep_alive) );
}

inline void set_opt_no_delay( nng::dialer_view s, bool v ) {
	s.set_opt_bool( to_name(option::no_delay), v );
}

inline void set_opt_keep_alive( nng::dialer_view s, bool v ) {
	s.set_opt_bool( to_name(option::keep_alive), v );
}

inline void set_opt_no_delay( nng::listener_view s, bool v ) {
	s.set_opt_bool( to_name(option::no_delay), v );
}

inline void set_opt_keep_alive( nng::listener_view s, bool v ) {
	s.set_opt_bool( to_name(option::keep_alive), v );
}

inline void set_opt_no_delay( nng::socket_view s, bool v ) {
	s.set_opt_bool( to_name(option::no_delay), v );
}

inline void set_opt_keep_alive( nng::socket_view s, bool v ) {
	s.set_opt_bool( to_name(option::keep_alive), v );
}

}}

#endif
