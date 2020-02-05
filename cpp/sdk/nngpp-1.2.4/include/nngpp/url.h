#ifndef NNGPP_URL_H
#define NNGPP_URL_H
#include "url_view.h"

namespace nng {

struct url : url_view {

	url() = default;

	explicit url( nng_url* u ) noexcept : url_view(u) {}
	
	explicit url( const char* str ) {
		int r = nng_url_parse( &u, str );
		if( r != 0 ) {
			throw exception(r,"nng_url_parse");
		}
	}
	
	url( const url& rhs ) : url_view() {
		int r = nng_url_clone(&u,rhs.get());
		if( r != 0 ) {
			throw exception(r,"nng_url_parse");
		}
	}
	
	url( url&& rhs ) noexcept : url_view(rhs.u) {
		rhs.u = nullptr;
	}
	
	url& operator=( const url& rhs ) {
		if( this != &rhs ) {
			if( u != nullptr ) nng_url_free(u);
			int r = nng_url_clone(&u,rhs.get());
			if( r != 0 ) {
				throw exception(r,"nng_url_clone");
			}
		}
		return *this;
	}
	
	url& operator=( url&& rhs ) noexcept {
		if( this != &rhs ) {
			if( u != nullptr ) nng_url_free(u);
			u = rhs.u;
			rhs.u = nullptr;
		}
		return *this;
	}
	
	~url() {
		if( u != nullptr ) nng_url_free(u);
	}
	
	nng_url* release() noexcept {
		auto out = u;
		u = nullptr;
		return out;
	}

};

inline url make_url( const char* str ) {
	return url(str);
}

}

#endif
