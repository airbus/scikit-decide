#ifndef NNGPP_STREAM_LISTENER_H
#define NNGPP_STREAM_LISTENER_H
#include "listener_view.h"
#include <nngpp/url_view.h>

namespace nng { namespace stream {

struct listener : listener_view {

	listener() = default;
	
	explicit listener( nng_stream_listener* d ) noexcept : listener_view(d) {}

	explicit listener( const char* url ) {
		int r = nng_stream_listener_alloc( &d, url );
		if( r != 0 ) {
			throw exception(r,"nng_stream_listener_alloc");
		}
	}

	explicit listener( nng::url_view url ) {
		int r = nng_stream_listener_alloc_url( &d, url.get() );
		if( r != 0 ) {
			throw exception(r,"nng_stream_listener_alloc_url");
		}
	}
	
	listener( const listener& rhs ) = delete;
	
	listener( listener&& rhs ) noexcept : listener_view(rhs.d) {
		rhs.d = nullptr;
	}
	
	listener& operator=( const listener& rhs ) = delete;
	
	listener& operator=( listener&& rhs ) noexcept {
		if( this != &rhs ) {
			if( d != nullptr ) nng_stream_listener_free(d);
			d = rhs.d;
			rhs.d = nullptr;
		}
		return *this;
	}
	
	~listener() {
		if( d != nullptr ) nng_stream_listener_free(d);
	}
	
	nng_stream_listener* release() noexcept {
		auto out = d;
		d = nullptr;
		return out;
	}

};

inline listener make_listener( const char* url ) {
	return listener(url);
}

inline listener make_listener( nng::url_view url ) {
	return listener(url);
}

}}

#endif
