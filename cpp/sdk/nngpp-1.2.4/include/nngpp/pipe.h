#ifndef NNGPP_PIPE_H
#define NNGPP_PIPE_H
#include "pipe_view.h"

namespace nng {

struct pipe : pipe_view {

	pipe() = default;

	explicit pipe( nng_pipe p ) noexcept : pipe_view(p) {}
	
	pipe( const pipe& rhs ) = delete;
	
	pipe( pipe&& rhs ) noexcept : pipe_view(rhs.p) {
		rhs.p.id = 0;
	}
	
	pipe& operator=( const pipe& rhs ) = delete;
	
	pipe& operator=( pipe&& rhs ) {
		if( this != &rhs ) {
			if( p.id != 0 ) {
				int r = nng_pipe_close(p);
				if( r != 0 ) {
					throw exception(r,"nng_pipe_close");
				}
			}
			p = rhs.p;
			rhs.p.id = 0;
		}
		return *this;
	}
	
	~pipe() {
		if( p.id != 0 ) nng_pipe_close(p);
	}

	nng_pipe release() noexcept {
		auto out = p;
		p.id = 0;
		return out;
	}

};

}

#endif
