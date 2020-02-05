#ifndef NNGPP_MSG_VIEW_H
#define NNGPP_MSG_VIEW_H
#include "msg_header.h"
#include "msg_body.h"
#include "pipe_view.h"

namespace nng {

struct msg_view {
protected:
	nng_msg* m = nullptr;
	
public:
	msg_view() = default;

	msg_view( nng_msg* m ) noexcept : m(m) {}
	
	nng_msg* get() const noexcept {
		return m;
	}

	nng_msg* operator->() const noexcept {
		return m;
	}

	explicit operator bool() const noexcept {
		return m != nullptr;
	}
	
	void realloc( size_t size ) const {
		int r = nng_msg_realloc(m,size);
		if( r != 0 ) {
			throw exception(r,"nng_msg_realloc");
		}
	}
	
	msg_header header() const noexcept {
		return msg_header(m);
	}
	
	msg_body body() const noexcept {
		return msg_body(m);
	}

	void set_pipe( pipe_view p ) const noexcept {
		nng_msg_set_pipe(m,p.get());
	}
	
	pipe_view get_pipe() const noexcept {
		return nng_msg_get_pipe(m);
	}
	
	size_t get_opt( int opt, buffer& out ) const {
		size_t size = out.size();
		int r = nng_msg_getopt(m,opt,out.data(),&size);
		if( r != 0 ) {
			throw exception(r,"nng_msg_getopt");
		}
		return size;
	}
};

}

#endif
