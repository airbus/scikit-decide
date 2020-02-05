#ifndef NNGPP_TRANSPORT_ZEROTIER_H
#define NNGPP_TRANSPORT_ZEROTIER_H
#include <nngpp/listener_view.h>
#include <nngpp/dialer_view.h>
#include <nngpp/pipe_view.h>
#include <nng/transport/zerotier/zerotier.h>

namespace nng { namespace zt {

inline void register_transport() {
	int r = nng_zt_register();
	if( r != 0 ) {
		throw exception(r,"nng_zt_register");
	}
}

enum class status {
	up        = NNG_ZT_STATUS_UP,
	config    = NNG_ZT_STATUS_CONFIG,
	denied    = NNG_ZT_STATUS_DENIED,
	not_found = NNG_ZT_STATUS_NOTFOUND,
	error     = NNG_ZT_STATUS_ERROR,
	obsolete  = NNG_ZT_STATUS_OBSOLETE,
	unknown   = NNG_ZT_STATUS_UNKNOWN
};

enum class option {
	mtu,            // size
	nwid,           // u64
	node,           // u64
	home,           // string
	network_status, // int
	network_name,   // string
	ping_time,      // ms
	ping_tries,     // int
	conn_time,      // ms
	conn_tries,     // int
	orbit,          // u64
	deorbit         // u64
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_ZT_MTU,
	NNG_OPT_ZT_NWID,
	NNG_OPT_ZT_NODE,
	NNG_OPT_ZT_HOME,
	NNG_OPT_ZT_NETWORK_STATUS,
	NNG_OPT_ZT_NETWORK_NAME,
	NNG_OPT_ZT_PING_TIME,
	NNG_OPT_ZT_PING_TRIES,
	NNG_OPT_ZT_CONN_TIME,
	NNG_OPT_ZT_CONN_TRIES,
	NNG_OPT_ZT_ORBIT,
	NNG_OPT_ZT_DEORBIT
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline size_t get_opt_mtu( nng::pipe_view s ) {
	return s.get_opt_size( to_name(option::mtu) );
}

inline uint64_t get_opt_nwid( nng::pipe_view s ) {
	return s.get_opt_uint64( to_name(option::nwid) );
}

inline uint64_t get_opt_node( nng::pipe_view s ) {
	return s.get_opt_uint64( to_name(option::node) );
}

inline buffer get_opt_home( nng::dialer_view s ) {
	return s.get_opt_string( to_name(option::home) );
}

inline void set_opt_home( nng::dialer_view s, const char* v ) {
	s.set_opt_string( to_name(option::home), v );
}

inline uint64_t get_opt_node( nng::dialer_view s ) {
	return s.get_opt_uint64( to_name(option::node) );
}

inline uint64_t get_opt_nwid( nng::dialer_view s ) {
	return s.get_opt_uint64( to_name(option::nwid) );
}

inline status get_opt_network_status( nng::dialer_view s ) {
	return (status)s.get_opt_int( to_name(option::network_status) );
}

inline buffer get_opt_network_name( nng::dialer_view s ) {
	return s.get_opt_string( to_name(option::network_name) );
}

inline nng_duration get_opt_ping_time( nng::dialer_view s ) {
	return s.get_opt_ms( to_name(option::ping_time) );
}

inline void set_opt_ping_time( nng::dialer_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::ping_time), v );
}

inline int get_opt_ping_tries( nng::dialer_view s ) {
	return s.get_opt_int( to_name(option::ping_tries) );
}

inline void set_opt_ping_tries( nng::dialer_view s, int v ) {
	s.set_opt_int( to_name(option::ping_tries), v );
}

inline nng_duration get_opt_conn_time( nng::dialer_view s ) {
	return s.get_opt_ms( to_name(option::conn_time) );
}

inline void set_opt_conn_time( dialer_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::conn_time), v );
}

inline int get_opt_conn_tries( dialer_view s ) {
	return s.get_opt_int( to_name(option::conn_tries) );
}

inline void set_opt_conn_tries( dialer_view s, int v ) {
	s.set_opt_int( to_name(option::conn_tries), v );
}

inline uint64_t get_opt_orbit( dialer_view s ) {
	return s.get_opt_uint64( to_name(option::orbit) );
}

inline void set_opt_orbit( dialer_view s, uint64_t v ) {
	s.set_opt_uint64( to_name(option::orbit), v );
}

inline uint64_t get_opt_deorbit( dialer_view s ) {
	return s.get_opt_uint64( to_name(option::deorbit) );
}

inline void set_opt_deorbit( dialer_view s, uint64_t v ) {
	s.set_opt_uint64( to_name(option::deorbit), v );
}

inline buffer get_opt_home( listener_view s ) {
	return s.get_opt_string( to_name(option::home) );
}

inline void set_opt_home( listener_view s, const char* v ) {
	s.set_opt_string( to_name(option::home), v );
}

inline uint64_t get_opt_node( listener_view s ) {
	return s.get_opt_uint64( to_name(option::node) );
}

inline uint64_t get_opt_nwid( listener_view s ) {
	return s.get_opt_uint64( to_name(option::nwid) );
}

inline status get_opt_network_status( listener_view s ) {
	return (status)s.get_opt_int( to_name(option::network_status) );
}

inline buffer get_opt_network_name( listener_view s ) {
	return s.get_opt_string( to_name(option::network_name) );
}

inline nng_duration get_opt_ping_time( listener_view s ) {
	return s.get_opt_ms( to_name(option::ping_time) );
}

inline void set_opt_ping_time( listener_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::ping_time), v );
}

inline int get_opt_ping_tries( listener_view s ) {
	return s.get_opt_int( to_name(option::ping_tries) );
}

inline void set_opt_ping_tries( listener_view s, int v ) {
	s.set_opt_int( to_name(option::ping_tries), v );
}

inline uint64_t get_opt_orbit( listener_view s ) {
	return s.get_opt_uint64( to_name(option::orbit) );
}

inline void set_opt_orbit( listener_view s, uint64_t v ) {
	s.set_opt_uint64( to_name(option::orbit), v );
}

inline uint64_t get_opt_deorbit( listener_view s ) {
	return s.get_opt_uint64( to_name(option::deorbit) );
}

inline void set_opt_deorbit( listener_view s, uint64_t v ) {
	s.set_opt_uint64( to_name(option::deorbit), v );
}

}}

#endif
