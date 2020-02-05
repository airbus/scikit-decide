#ifndef NNGPP_CTX_H
#define NNGPP_CTX_H
#include "ctx_view.h"
#include "socket_view.h"

namespace nng {

struct ctx : ctx_view {

	ctx() = default;
	
	explicit ctx( nng_ctx c ) noexcept : ctx_view(c) {}
	
	explicit ctx( socket_view s ) {
		int r = nng_ctx_open( &c, s.get() );
		if( r != 0 ) {
			throw exception(r,"nng_ctx_open");
		}
	}
	
	ctx( const ctx& rhs ) = delete;
	
	ctx( ctx&& rhs ) noexcept : ctx_view(rhs.c) {
		rhs.c.id = 0;
	}
	
	ctx& operator=( const ctx& rhs ) = delete;
	
	ctx& operator=( ctx&& rhs ) {
		if( this != &rhs ) {
			if( c.id != 0 ) {
				int r = nng_ctx_close(c);
				if( r != 0 && r != (int)nng::error::closed ) {
					throw exception(r,"nng_ctx_close");
				}
			}
			c = rhs.c;
			rhs.c.id = 0;
		}
		return *this;
	}
	
	~ctx() {
		if( c.id != 0 ) nng_ctx_close(c);
	}
	
	nng_ctx release() noexcept {
		auto out = c;
		c.id = 0;
		return out;
	}
	
};

inline ctx make_ctx( socket_view s ) {
	return ctx(s);
}

}

#endif
