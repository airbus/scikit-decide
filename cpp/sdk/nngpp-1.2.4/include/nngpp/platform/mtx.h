#ifndef NNGPP_MTX_H
#define NNGPP_MTX_H
#include "mtx_view.h"
#include <nngpp/error.h>

namespace nng {

struct mtx : mtx_view {

	mtx() = default;
	
	explicit mtx( nng_mtx* m ) noexcept : mtx_view(m) {}
	
	mtx( const mtx& rhs ) = delete;
	
	mtx( mtx&& rhs ) noexcept : mtx_view(rhs.m) {
		rhs.m = nullptr;
	}
	
	mtx& operator=( const mtx& rhs ) = delete;
	
	mtx& operator=( mtx&& rhs ) noexcept {
		if( this != &rhs ) {
			if( m != nullptr ) nng_mtx_free(m);
			m = rhs.m;
			rhs.m = nullptr;
		}
		return *this;
	}
	
	~mtx() {
		if( m != nullptr ) nng_mtx_free(m);
	}
	
	nng_mtx* release() noexcept {
		auto out = m;
		m = nullptr;
		return out;
	}

};

inline mtx make_mtx() {
	nng_mtx* m;
	int r = nng_mtx_alloc( &m );
	if( r != 0 ) {
		throw exception(r,"nng_mtx_alloc");
	}
	return mtx(m);
}

}

#endif
