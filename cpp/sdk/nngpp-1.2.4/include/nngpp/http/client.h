#ifndef NNGPP_HTTP_CLIENT_H
#define NNGPP_HTTP_CLIENT_H
#include "client_view.h"
#include <nngpp/url_view.h>

namespace nng { namespace http {

struct client : client_view {

	client() = default;
	
	explicit client( nng_http_client* c ) noexcept : client_view(c) {}
	
	explicit client( url_view u ) {
		int r = nng_http_client_alloc( &c, u.get() );
		if( r != 0 ) {
			throw exception(r,"nng_http_client_alloc");
		}
	}
	
	client( const client& rhs ) = delete;
	
	client( client&& rhs ) noexcept : client_view(rhs.c) {
		rhs.c = nullptr;
	}
	
	client& operator=( const client& rhs ) = delete;
	
	client& operator=( client&& rhs ) noexcept {
		if( this != &rhs ) {
			if( c != nullptr ) nng_http_client_free(c);
			c = rhs.c;
			rhs.c = nullptr;
		}
		return *this;
	}
	
	~client() {
		if( c != nullptr ) nng_http_client_free(c);
	}
	
	nng_http_client* release() noexcept {
		auto out = c;
		c = nullptr;
		return out;
	}

};

inline client make_client( url_view u ) {
	return client(u);
}

}}

#endif
