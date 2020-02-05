#ifndef NNGPP_HTTP_SERVER_H
#define NNGPP_HTTP_SERVER_H
#include "server_view.h"
#include <nngpp/url_view.h>

namespace nng { namespace http {

struct server : server_view {

	server() = default;
	
	explicit server( nng_http_server* s ) noexcept : server_view(s) {}
	
	explicit server( url_view u ) {
		int r = nng_http_server_hold( &s, u.get() );
		if( r != 0 ) {
			throw exception(r,"nng_http_server_hold");
		}
	}
	
	server( const server& rhs ) = delete;
	
	server( server&& rhs ) noexcept : server_view(rhs.s) {
		rhs.s = nullptr;
	}
	
	server& operator=( const server& rhs ) = delete;
	
	server& operator=( server&& rhs ) noexcept {
		if( this != &rhs ) {
			if( s != nullptr ) nng_http_server_release(s);
			s = rhs.s;
			rhs.s = nullptr;
		}
		return *this;
	}
	
	~server() {
		if( s != nullptr ) nng_http_server_release(s);
	}
	
	nng_http_server* release() noexcept {
		auto out = s;
		s = nullptr;
		return out;
	}

};

inline server make_server( url_view u ) {
	return server(u);
}

}}

#endif
