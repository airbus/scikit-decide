#ifndef NNGPP_LISTENER_H
#define NNGPP_LISTENER_H
#include "listener_view.h"
#include "socket_view.h"

namespace nng {

struct listener : listener_view {

	listener() = default;
	
	explicit listener( nng_listener lis ) noexcept : listener_view(lis) {}
	
	explicit listener( socket_view s, const char* addr, int flags ) {
		int r = nng_listen( s.get(), addr, &lis, flags );
		if( r != 0 ) {
			throw exception(r,"nng_listen");
		}
	}
	
	explicit listener( socket_view s, const char* addr ) {
		int r = nng_listener_create( &lis, s.get(), addr );
		if( r != 0 ) {
			throw exception(r,"nng_listener_create");
		}
	}
	
	listener( const listener& rhs ) = delete;
	
	listener( listener&& rhs ) noexcept : listener_view(rhs.lis) {
		rhs.lis.id = 0;
	}
	
	listener& operator=( const listener& rhs ) = delete;
	
	listener& operator=( listener&& rhs ) {
		if( this != &rhs ) {
			if( lis.id != 0 ) {
				int r = nng_listener_close(lis);
				if( r != 0 && r != (int)nng::error::closed ) {
					throw exception(r,"nng_listener_close");
				}
			}
			lis = rhs.lis;
			rhs.lis.id = 0;
		}
		return *this;
	}
	
	~listener() {
		if( lis.id != 0 ) nng_listener_close(lis);
	}

	nng_listener release() noexcept {
		auto out = lis;
		lis.id = 0;
		return out;
	}

};

inline listener make_listener( socket_view s, const char* addr, int flags ) {
	return listener(s,addr,flags);
}

inline listener make_listener( socket_view s, const char* addr ) {
	return listener(s,addr);
}

}

#endif
