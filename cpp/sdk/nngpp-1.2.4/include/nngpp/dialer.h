#ifndef NNGPP_DIALER_H
#define NNGPP_DIALER_H
#include "dialer_view.h"
#include "socket_view.h"

namespace nng {

struct dialer : dialer_view {

	dialer() = default;
	
	explicit dialer( nng_dialer d ) noexcept : dialer_view(d) {}
	
	explicit dialer( socket_view s, const char* addr, int flags ) {
		int r = nng_dial( s.get(), addr, &d, flags );
		if( r != 0 ) {
			throw exception(r,"nng_dial");
		}
	}
	
	explicit dialer( socket_view s, const char* addr ) {
		int r = nng_dialer_create( &d, s.get(), addr );
		if( r != 0 ) {
			throw exception(r,"nng_dialer_create");
		}
	}
	
	dialer( const dialer& rhs ) = delete;
	
	dialer( dialer&& rhs ) noexcept : dialer_view(rhs.d) {
		rhs.d.id = 0;
	}
	
	dialer& operator=( const dialer& rhs ) = delete;
	
	dialer& operator=( dialer&& rhs ) {
		if( this != &rhs ) {
			if( d.id != 0 ) {
				int r = nng_dialer_close(d);
				if( r != 0 && r != (int)nng::error::closed ) {
					throw exception(r,"nng_dialer_close");
				}
			}
			d = rhs.d;
			rhs.d.id = 0;
		}
		return *this;
	}
	
	~dialer() {
		if( d.id != 0 ) nng_dialer_close(d);
	}
	
	nng_dialer release() noexcept {
		auto out = d;
		d.id = 0;
		return out;
	}

};

inline dialer make_dialer( socket_view s, const char* addr, int flags ) {
	return dialer(s,addr,flags);
}

inline dialer make_dialer( socket_view s, const char* addr ) {
	return dialer(s,addr);
}

}

#endif
