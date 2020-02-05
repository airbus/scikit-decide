#ifndef NNGPP_THREAD_H
#define NNGPP_THREAD_H
#include "thread_view.h"
#include <nngpp/error.h>

namespace nng {

struct thread : thread_view {

	thread() = default;
	
	explicit thread( nng_thread* t ) noexcept : thread_view(t) {}

	explicit thread( void(*f)(void*), void* arg ) {
		int r = nng_thread_create( &t, f, arg );
		if( r != 0 ) {
			throw exception(r,"nng_thread_create");
		}
	}

	thread( const thread& rhs ) = delete;
	
	thread( thread&& rhs ) noexcept : thread_view(rhs.t) {
		rhs.t = nullptr;
	}
	
	thread& operator=( const thread& rhs ) = delete;
	
	thread& operator=( thread&& rhs ) noexcept {
		if( this != &rhs ) {
			if( t != nullptr ) nng_thread_destroy(t);
			t = rhs.t;
			rhs.t = nullptr;
		}
		return *this;
	}
	
	~thread() {
		if( t != nullptr ) nng_thread_destroy(t);
	}
	
	nng_thread* release() noexcept {
		auto out = t;
		t = nullptr;
		return out;
	}

};

inline thread make_thread( void(*f)(void*), void* arg ) {
	return thread(f,arg);
}

}

#endif
