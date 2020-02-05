#ifndef NNGPP_SOCKET_H
#define NNGPP_SOCKET_H
#include "socket_view.h"

namespace nng {

struct socket : socket_view {

	socket() = default;

	explicit socket( nng_socket s ) noexcept : socket_view(s) {}
	
	socket( const socket& rhs ) = delete;
	
	socket( socket&& rhs ) noexcept : socket_view(rhs.s) {
		rhs.s.id = 0;
	}
	
	socket& operator=( const socket& rhs ) = delete;
	
	socket& operator=( socket&& rhs ) {
		if( this != &rhs ) {
			if( s.id != 0 ) {
				int r = nng_close(s);
				if( r != 0 && r != (int)nng::error::closed ) {
					throw exception(r,"nng_close");
				}
			}
			s = rhs.s;
			rhs.s.id = 0;
		}
		return *this;
	}
	
	~socket() {
		if( s.id != 0 ) nng_close(s);
	}
	
	nng_socket release() noexcept {
		auto out = s;
		s.id = 0;
		return out;
	}

};

}

#endif
