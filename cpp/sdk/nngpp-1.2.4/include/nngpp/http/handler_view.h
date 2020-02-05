#ifndef NNGPP_HTTP_HANDLER_VIEW_H
#define NNGPP_HTTP_HANDLER_VIEW_H
#include <nngpp/error.h>
#include "misc.h"

namespace nng { namespace http {

struct handler_view {
protected:
	nng_http_handler* h = nullptr;
	
public:
	handler_view() = default;
	
	handler_view( nng_http_handler* h ) noexcept : h(h) {}
	
	nng_http_handler* get() const noexcept {
		return h;
	}

	nng_http_handler* operator->() const noexcept {
		return h;
	}

	explicit operator bool() const noexcept {
		return h != nullptr;
	}
	
	void set_method( const char* method ) const {
		int r = nng_http_handler_set_method(h,method);
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_set_method");
		}
	}

	void set_method( verb v ) const {
		set_method( to_name(v) );
	}

	void set_host( const char* host ) const {
		int r = nng_http_handler_set_host(h,host);
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_set_host");
		}
	}

	void set_tree() const {
		int r = nng_http_handler_set_tree(h);
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_set_tree");
		}
	}
	
	void* get_data() const noexcept {
		return nng_http_handler_get_data(h);
	}
	
	void set_data( void* data, void (*dtor)(void*) ) const {
		int r = nng_http_handler_set_data(h,data,dtor);
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_set_data");
		}
	}

	void collect_body( bool want, size_t maxsz ) const {
		int r = nng_http_handler_collect_body(h,want,maxsz);
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_collect_body");
		}
	}
};

}}

#endif
