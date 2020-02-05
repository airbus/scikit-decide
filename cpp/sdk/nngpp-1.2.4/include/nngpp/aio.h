#ifndef NNGPP_AIO_H
#define NNGPP_AIO_H
#include "aio_view.h"

namespace nng {

struct aio : aio_view {

	aio() = default;

	explicit aio( nng_aio* a ) noexcept : aio_view(a) {}
	
	explicit aio( void (*cb)(void*), void* arg ) {
		int r = nng_aio_alloc( &a, cb, arg );
		if( r != 0 ) {
			throw exception(r,"nng_aio_alloc");
		}
	}
	
	aio( const aio& rhs ) = delete;
	
	aio( aio&& rhs ) noexcept : aio_view(rhs.a) {
		rhs.a = nullptr;
	}
	
	aio& operator=( const aio& rhs ) = delete;
	
	aio& operator=( aio&& rhs ) noexcept {
		if( this != &rhs ) {
			if( a != nullptr ) nng_aio_free(a);
			a = rhs.a;
			rhs.a = nullptr;
		}
		return *this;
	}
	
	~aio() {
		if( a != nullptr ) nng_aio_free(a);
	}

	nng_aio* release() noexcept {
		auto out = a;
		a = nullptr;
		return out;
	}

};

inline aio make_aio( void (*cb)(void*), void* arg ) {
	return aio(cb,arg);
}

inline aio make_aio() {
	return aio(nullptr,nullptr);
}

}

#endif
