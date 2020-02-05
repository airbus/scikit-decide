#ifndef NNGPP_BUFFER_H
#define NNGPP_BUFFER_H
#include "view.h"

namespace nng {

struct buffer : view {

	buffer() = default;

	template<typename T>
	explicit buffer( T* d, size_t s ) noexcept : view((pointer)d,s) {}

	explicit buffer( size_t sz ) : view(nng_alloc(sz),sz) {
		if( d == nullptr ) {
			s = 0;
			throw exception(NNG_ENOMEM,"nng_alloc");
		}
	}
	
	buffer( const buffer& rhs ) : buffer(rhs.s) {
		memcpy(d,rhs.d,s);
	}
	
	buffer( buffer&& rhs ) noexcept : view(rhs.d,rhs.s) {
		rhs.d = nullptr;
		rhs.s = 0;
	}
	
	buffer& operator=( const buffer& rhs ) {
		if( this != &rhs ) {
			if( s != rhs.s ) {
				if( d != nullptr ) nng_free(d,s);
				d = nng_alloc(rhs.s);
				s = rhs.s;
				if( d == nullptr ) {
					s = 0;
					throw exception(NNG_ENOMEM,"nng_alloc");
				}
			}
			memcpy(d,rhs.d,s);
		}
		return *this;
	}
	
	buffer& operator=( buffer&& rhs ) noexcept {
		if( this != &rhs ) {
			if( d != nullptr ) nng_free(d,s);
			d = rhs.d;
			s = rhs.s;
			rhs.d = nullptr;
			rhs.s = 0;
		}
		return *this;
	}
	
	~buffer() {
		if( d != nullptr ) nng_free(d,s);
	}

	pointer release() noexcept {
		auto out = d;
		d = nullptr;
		s = 0;
		return out;
	}

};

inline buffer make_buffer( size_t sz ) {
	return buffer(sz);
}

}

#endif
