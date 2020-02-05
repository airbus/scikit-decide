#ifndef NNGPP_STREAM_DIALER_H
#define NNGPP_STREAM_DIALER_H
#include "dialer_view.h"
#include <nngpp/url_view.h>

namespace nng { namespace stream {

struct dialer : dialer_view {

	dialer() = default;
	
	explicit dialer( nng_stream_dialer* d ) noexcept : dialer_view(d) {}

	explicit dialer( const char* url ) {
		int r = nng_stream_dialer_alloc( &d, url );
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_alloc");
		}
	}

	explicit dialer( nng::url_view url ) {
		int r = nng_stream_dialer_alloc_url( &d, url.get() );
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_alloc_url");
		}
	}

	dialer( const dialer& rhs ) = delete;
	
	dialer( dialer&& rhs ) noexcept : dialer_view(rhs.d) {
		rhs.d = nullptr;
	}
	
	dialer& operator=( const dialer& rhs ) = delete;
	
	dialer& operator=( dialer&& rhs ) noexcept {
		if( this != &rhs ) {
			if( d != nullptr ) nng_stream_dialer_free(d);
			d = rhs.d;
			rhs.d = nullptr;
		}
		return *this;
	}
	
	~dialer() {
		if( d != nullptr ) nng_stream_dialer_free(d);
	}
	
	nng_stream_dialer* release() noexcept {
		auto out = d;
		d = nullptr;
		return out;
	}

};

inline dialer make_dialer( const char* url ) {
	return dialer(url);
}

inline dialer make_dialer( nng::url_view url ) {
	return dialer(url);
}

}}

#endif
