#ifndef NNGPP_HTTP_HANDLER_H
#define NNGPP_HTTP_HANDLER_H
#include "handler_view.h"

namespace nng { namespace http {

struct path_type_file {};
struct path_type_dir {};

struct handler : handler_view {

	handler() = default;
	
	explicit handler( nng_http_handler* h ) noexcept : handler_view(h) {}
	
	explicit handler( const char* uri, void (*cb)(nng_aio*) ) {
		int r = nng_http_handler_alloc( &h, uri, cb );
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_alloc");
		}
	}
	
	explicit handler( const char* uri, const char* path, path_type_file ) {
		int r = nng_http_handler_alloc_file( &h, uri, path );
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_alloc_file");
		}
	}
	
	explicit handler( const char* uri, const char* path, path_type_dir ) {
		int r = nng_http_handler_alloc_directory( &h, uri, path );
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_alloc_directory");
		}
	}
	
	explicit handler( const char* uri, const void* data, size_t size, const char* ctype ) {
		int r = nng_http_handler_alloc_static( &h, uri, data, size, ctype );
		if( r != 0 ) {
			throw exception(r,"nng_http_handler_alloc_static");
		}
	}
	
	handler( const handler& rhs ) = delete;
	
	handler( handler&& rhs ) noexcept : handler_view(rhs.h) {
		rhs.h = nullptr;
	}
	
	handler& operator=( const handler& rhs ) = delete;
	
	handler& operator=( handler&& rhs ) noexcept {
		if( this != &rhs ) {
			if( h != nullptr ) nng_http_handler_free(h);
			h = rhs.h;
			rhs.h = nullptr;
		}
		return *this;
	}
	
	~handler() {
		if( h != nullptr ) nng_http_handler_free(h);
	}
	
	nng_http_handler* release() noexcept {
		auto out = h;
		h = nullptr;
		return out;
	}

};

inline handler make_handler( const char* uri, void (*cb)(nng_aio*) ) {
	return handler(uri,cb);
}

inline handler make_handler( const char* uri, const char* path, path_type_file path_type ) {
	return handler(uri,path,path_type);
}

inline handler make_handler( const char* uri, const char* path, path_type_dir path_type ) {
	return handler(uri,path,path_type);
}

inline handler make_handler( const char* uri, const void* data, size_t size, const char* ctype ) {
	return handler(uri,data,size,ctype);
}

}}

#endif
