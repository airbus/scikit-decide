#ifndef NNGPP_HTTP_REQ_H
#define NNGPP_HTTP_REQ_H
#include "req_view.h"
#include <nngpp/url_view.h>

namespace nng { namespace http {

struct req : req_view {

	req() = default;

	explicit req( nng_http_req* q ) noexcept : req_view(q) {}
	
	explicit req( url_view u ) {
		int r = nng_http_req_alloc( &q, u.get() );
		if( r != 0 ) {
			throw exception(r,"nng_http_req_alloc");
		}
	}
	
	req( const req& rhs ) = delete;
	
	req( req&& rhs ) noexcept : req_view(rhs.q) {
		rhs.q = nullptr;
	}
	
	req& operator=( const req& rhs ) = delete;
	
	req& operator=( req&& rhs ) noexcept {
		if( this != &rhs ) {
			if( q != nullptr ) nng_http_req_free(q);
			q = rhs.q;
			rhs.q = nullptr;
		}
		return *this;
	}
	
	~req() {
		if( q != nullptr ) nng_http_req_free(q);
	}

	nng_http_req* release() noexcept {
		auto out = q;
		q = nullptr;
		return out;
	}

};

inline req make_req( url_view u ) {
	return req(u);
}

}}

#endif
