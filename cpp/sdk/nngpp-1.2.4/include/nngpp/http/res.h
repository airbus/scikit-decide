#ifndef NNGPP_HTTP_RES_H
#define NNGPP_HTTP_RES_H
#include "res_view.h"

namespace nng { namespace http {

struct res : res_view {

	res() = default;
	
	explicit res( nng_http_res* s ) noexcept : res_view(s) {}
	
	explicit res( status code ) {
		int r = nng_http_res_alloc_error( &s, (uint16_t)code );
		if( r != 0 ) {
			throw exception(r,"nng_http_res_alloc_error");
		}
	}
	
	res( const res& rhs ) = delete;
	
	res( res&& rhs ) noexcept : res_view(rhs.s) {
		rhs.s = nullptr;
	}
	
	res& operator=( const res& rhs ) = delete;
	
	res& operator=( res&& rhs ) noexcept {
		if( this != &rhs ) {
			if( s != nullptr ) nng_http_res_free(s);
			s = rhs.s;
			rhs.s = nullptr;
		}
		return *this;
	}
	
	~res() {
		if( s != nullptr ) nng_http_res_free(s);
	}
	
	nng_http_res* release() noexcept {
		auto out = s;
		s = nullptr;
		return out;
	}

};

inline res make_res() {
	nng_http_res* s;
	int r = nng_http_res_alloc( &s );
	if( r != 0 ) {
		throw exception(r,"nng_http_res_alloc");
	}
	return res(s);
}

inline res make_res( status code ) {
	return res(code);
}

}}

#endif
