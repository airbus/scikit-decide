#ifndef NNGPP_HTTP_CONN_H
#define NNGPP_HTTP_CONN_H
#include "conn_view.h"

namespace nng { namespace http {

struct conn : conn_view {

	conn() = default;

	explicit conn( nng_http_conn* n ) noexcept : conn_view(n) {}
	
	conn( const conn& rhs ) = delete;
	
	conn( conn&& rhs ) noexcept : conn_view(rhs.n) {
		rhs.n = nullptr;
	}
	
	conn& operator=( const conn& rhs ) = delete;
	
	conn& operator=( conn&& rhs ) noexcept {
		if( this != &rhs ) {
			if( n != nullptr ) nng_http_conn_close(n);
			n = rhs.n;
			rhs.n = nullptr;
		}
		return *this;
	}
	
	~conn() {
		if( n != nullptr ) nng_http_conn_close(n);
	}
	
	nng_http_conn* release() noexcept {
		auto out = n;
		n = nullptr;
		return out;
	}

};

}}

#endif
