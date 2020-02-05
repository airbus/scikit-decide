#ifndef NNGPP_MSG_H
#define NNGPP_MSG_H
#include "msg_view.h"

namespace nng {

struct msg : msg_view {

	msg() = default;

	explicit msg( nng_msg* m ) noexcept : msg_view(m) {}
	
	explicit msg( size_t size ) {
		int r = nng_msg_alloc( &m, size );
		if( r != 0 ) {
			throw exception(r,"nng_msg_alloc");
		}
	}
	
	msg( const msg& rhs ) : msg_view() {
		int r = nng_msg_dup(&m,rhs.get());
		if( r != 0 ) {
			throw exception(r,"nng_msg_dup");
		}
	}
	
	msg( msg&& rhs ) noexcept : msg_view(rhs.m) {
		rhs.m = nullptr;
	}
	
	msg& operator=( const msg& rhs ) {
		if( this != &rhs ) {
			if( m != nullptr ) nng_msg_free(m);
			int r = nng_msg_dup(&m,rhs.get());
			if( r != 0 ) {
				throw exception(r,"nng_msg_dup");
			}
		}
		return *this;
	}
	
	msg& operator=( msg&& rhs ) noexcept {
		if( this != &rhs ) {
			if( m != nullptr ) nng_msg_free(m);
			m = rhs.m;
			rhs.m = nullptr;
		}
		return *this;
	}
	
	~msg() {
		if( m != nullptr ) nng_msg_free(m);
	}
	
	nng_msg* release() noexcept {
		auto out = m;
		m = nullptr;
		return out;
	}

};

inline msg make_msg( size_t size ) {
	return msg(size);
}

}

#endif
