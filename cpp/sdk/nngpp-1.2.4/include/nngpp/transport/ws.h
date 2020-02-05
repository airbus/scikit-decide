#ifndef NNGPP_TRANSPORT_WS_H
#define NNGPP_TRANSPORT_WS_H
#include <nngpp/listener_view.h>
#include <nngpp/dialer_view.h>
#include <nngpp/pipe_view.h>
#include <nng/transport/ws/websocket.h>

namespace nng { namespace ws {

inline void register_transport() {
	int r = nng_ws_register();
	if( r != 0 ) {
		throw exception(r,"nng_ws_register");
	}
}

enum class option {
	request_headers,  // string
	response_headers, // string
	request_header,   // string
	response_header,  // string
	request_uri,      // string
	send_frame_max,   // size
	recv_frame_max,   // size
	protocol,         // string
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_WS_REQUEST_HEADERS,
	NNG_OPT_WS_RESPONSE_HEADERS,
	NNG_OPT_WS_REQUEST_HEADER,
	NNG_OPT_WS_RESPONSE_HEADER,
	NNG_OPT_WS_REQUEST_URI,
	NNG_OPT_WS_SENDMAXFRAME,
	NNG_OPT_WS_RECVMAXFRAME,
	NNG_OPT_WS_PROTOCOL
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline buffer get_opt_request_headers( nng::pipe_view s ) {
	return s.get_opt_string( to_name(option::request_headers) );
}

inline buffer get_opt_response_headers( nng::pipe_view s ) {
	return s.get_opt_string( to_name(option::response_headers) );
}

inline buffer get_opt_request_header( nng::pipe_view s ) {
	return s.get_opt_string( to_name(option::request_header) );
}

inline buffer get_opt_response_header( nng::pipe_view s ) {
	return s.get_opt_string( to_name(option::response_header) );
}

inline buffer get_opt_request_uri( nng::pipe_view s ) {
	return s.get_opt_string( to_name(option::request_uri) );
}

inline void set_opt_request_headers( nng::dialer_view s, const char* v ) {
	s.set_opt_string( to_name(option::request_headers), v );
}

inline void set_opt_response_headers( nng::listener_view s, const char* v ) {
	s.set_opt_string( to_name(option::response_headers), v );
}

inline void set_opt_request_header( nng::dialer_view s, const char* v ) {
	s.set_opt_string( to_name(option::request_header), v );
}

inline void set_opt_response_header( nng::listener_view s, const char* v ) {
	s.set_opt_string( to_name(option::response_header), v );
}

inline void set_opt_request_uri( nng::dialer_view s, const char* v ) {
	s.set_opt_string( to_name(option::request_uri), v );
}

}}

#endif
