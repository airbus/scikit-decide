#ifndef NNGPP_STREAM_STREAM_H
#define NNGPP_STREAM_STREAM_H
#include "stream_view.h"
#include "dialer.h"
#include "listener.h"

namespace nng { namespace stream {

struct stream : stream_view {

	stream() = default;
	
	explicit stream( nng_stream* s ) noexcept : stream_view(s) {}
	
	stream( const stream& rhs ) = delete;
	
	stream( stream&& rhs ) noexcept : stream_view(rhs.s) {
		rhs.s = nullptr;
	}
	
	stream& operator=( const stream& rhs ) = delete;
	
	stream& operator=( stream&& rhs ) noexcept {
		if( this != &rhs ) {
			if( s != nullptr ) nng_stream_free(s);
			s = rhs.s;
			rhs.s = nullptr;
		}
		return *this;
	}
	
	~stream() {
		if( s != nullptr ) nng_stream_free(s);
	}
	
	nng_stream* release() noexcept {
		auto out = s;
		s = nullptr;
		return out;
	}

};

}}

#endif
