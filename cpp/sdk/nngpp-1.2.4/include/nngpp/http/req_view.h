#ifndef NNGPP_HTTP_REQ_VIEW_H
#define NNGPP_HTTP_REQ_VIEW_H
#include <nngpp/view.h>
#include "misc.h"

namespace nng { namespace http {

struct req_view {
protected:
	nng_http_req* q = nullptr;
	
public:
	req_view() = default;

	req_view( nng_http_req* q ) noexcept : q(q) {}

	nng_http_req* get() const noexcept {
		return q;
	}

	nng_http_req* operator->() const noexcept {
		return q;
	}

	explicit operator bool() const noexcept {
		return q != nullptr;
	}
	
	void reset() const noexcept {
		nng_http_req_reset(q);
	}

	const char* get_method() const noexcept {
		return nng_http_req_get_method(q);
	}
	
	void set_method( const char* method ) const {
		int r = nng_http_req_set_method(q,method);
		if( r != 0 ) {
			throw exception(r,"nng_http_req_set_method");
		}
	}
	
	const char* get_version() const noexcept {
		return nng_http_req_get_version(q);
	}
	
	void set_version( const char* version ) const {
		int r = nng_http_req_set_version(q,version);
		if( r != 0 ) {
			throw exception(r,"nng_http_req_set_version");
		}
	}
	
	const char* get_uri() const noexcept {
		return nng_http_req_get_uri(q);
	}
	
	void set_uri( const char* uri ) const {
		int r = nng_http_req_set_uri(q,uri);
		if( r != 0 ) {
			throw exception(r,"nng_http_req_set_uri");
		}
	}
	
	const char* get_header( const char* name ) const noexcept {
		return nng_http_req_get_header(q,name);
	}
	
	void set_header( const char* name, const char* value ) const {
		int r = nng_http_req_set_header(q,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_http_req_set_header");
		}
	}
	
	void add_header( const char* name, const char* value ) const {
		int r = nng_http_req_add_header(q,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_http_req_add_header");
		}
	}
	
	void del_header( const char* name ) const {
		int r = nng_http_req_del_header(q,name);
		if( r != 0 ) {
			throw exception(r,"nng_http_req_del_header");
		}
	}

	view get_data() const noexcept {
		void* data;
		size_t size;
		nng_http_req_get_data(q,&data,&size);
		return view(data,size);
	}
	
	void set_data( view v ) const {
		int r = nng_http_req_set_data(q,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_http_req_set_data");
		}
	}
	
	void copy_data( view v ) const {
		int r = nng_http_req_copy_data(q,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_http_req_copy_data");
		}
	}

};

}}

#endif
