#ifndef NNGPP_HTTP_RES_VIEW_H
#define NNGPP_HTTP_RES_VIEW_H
#include <nngpp/view.h>
#include "misc.h"

namespace nng { namespace http {

struct res_view {
protected:
	nng_http_res* s = nullptr;
	
public:
	res_view() = default;

	res_view( nng_http_res* s ) noexcept : s(s) {}
	
	nng_http_res* get() const noexcept {
		return s;
	}

	nng_http_res* operator->() const noexcept {
		return s;
	}

	explicit operator bool() const noexcept {
		return s != nullptr;
	}
	
	void reset() const noexcept {
		nng_http_res_reset(s);
	}

	status get_status() const noexcept {
		return (status)nng_http_res_get_status(s);
	}
	
	void set_status( status status ) const {
		int r = nng_http_res_set_status(s,(uint16_t)status);
		if( r != 0 ) {
			throw exception(r,"nng_http_res_set_status");
		}
	}
	
	const char* get_reason() const noexcept {
		return nng_http_res_get_reason(s);
	}
	
	void set_reason( const char* reason ) const {
		int r = nng_http_res_set_reason(s,reason);
		if( r != 0 ) {
			throw exception(r,"nng_http_res_set_reason");
		}
	}
	
	const char* get_header( const char* name ) const noexcept {
		return nng_http_res_get_header(s,name);
	}
	
	void set_header( const char* name, const char* value ) const {
		int r = nng_http_res_set_header(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_http_res_set_header");
		}
	}
	
	void add_header( const char* name, const char* value ) const {
		int r = nng_http_res_add_header(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_http_res_add_header");
		}
	}
	
	void del_header( const char* name ) const {
		int r = nng_http_res_del_header(s,name);
		if( r != 0 ) {
			throw exception(r,"nng_http_res_del_header");
		}
	}
	
	const char* get_version() const noexcept {
		return nng_http_res_get_version(s);
	}
	
	void set_version( const char* version ) const {
		int r = nng_http_res_set_version(s,version);
		if( r != 0 ) {
			throw exception(r,"nng_http_res_set_version");
		}
	}

	view get_data() const noexcept {
		void* data;
		size_t size;
		nng_http_res_get_data(s,&data,&size);
		return view(data,size);
	}

	void set_data( view v ) const {
		int r = nng_http_res_set_data(s,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_http_res_set_data");
		}
	}
	
	void copy_data( view v ) const {
		int r = nng_http_res_copy_data(s,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_http_res_copy_data");
		}
	}
};

}}

#endif
