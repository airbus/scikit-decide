#ifndef NNGPP_CV_H
#define NNGPP_CV_H
#include "cv_view.h"
#include "mtx_view.h"

namespace nng {

struct cv : cv_view {

	cv() = default;
	
	explicit cv( nng_cv* c ) noexcept : cv_view(c) {}
	
	explicit cv( mtx_view mut ) {
		int r = nng_cv_alloc( &c, mut.get() );
		if( r != 0 ) {
			throw exception(r,"nng_cv_alloc");
		}
	}
	
	cv( const cv& rhs ) = delete;
	
	cv( cv&& rhs ) noexcept : cv_view(rhs.c) {
		rhs.c = nullptr;
	}
	
	cv& operator=( const cv& rhs ) = delete;
	
	cv& operator=( cv&& rhs ) noexcept {
		if( this != &rhs ) {
			if( c != nullptr ) nng_cv_free(c);
			c = rhs.c;
			rhs.c = nullptr;
		}
		return *this;
	}
	
	~cv() {
		if( c != nullptr ) nng_cv_free(c);
	}
	
	nng_cv* release() noexcept {
		auto out = c;
		c = nullptr;
		return out;
	}

};

inline cv make_cv( mtx_view mut ) {
	return cv(mut);
}

}

#endif
