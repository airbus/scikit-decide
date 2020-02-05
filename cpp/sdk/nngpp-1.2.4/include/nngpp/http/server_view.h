#ifndef NNGPP_HTTP_SERVER_VIEW_H
#define NNGPP_HTTP_SERVER_VIEW_H
#include <nngpp/transport/tls/config_view.h>
#include "handler.h"
#include "res_view.h"

namespace nng { namespace http {

struct server_view {
protected:
	nng_http_server* s = nullptr;
	
public:
	server_view() = default;

	server_view( nng_http_server* s ) noexcept : s(s) {}
	
	nng_http_server* get() const noexcept {
		return s;
	}

	nng_http_server* operator->() const noexcept {
		return s;
	}

	explicit operator bool() const noexcept {
		return s != nullptr;
	}
		
	void start() const {
		int r = nng_http_server_start(s);
		if( r != 0 ) {
			throw exception(r,"nng_http_server_start");
		}
	}
		
	void stop() const noexcept {
		nng_http_server_stop(s);
	}
	
	void add_handler( handler&& h ) const {
		int r = nng_http_server_add_handler(s,h.get());
		if( r != 0 ) {
			throw exception(r,"nng_http_server_add_handler");
		}
		// if successful, the handler is owned by the server
		h.release();
	}
	
	handler remove_handler( handler_view h ) const {
		int r = nng_http_server_del_handler(s,h.get());
		if( r != 0 ) {
			throw exception(r,"nng_http_server_del_handler");
		}
		return handler(h.get());
	}
	
	tls::config_view get_tls() const {
		nng_tls_config* cfg;
		int r = nng_http_server_get_tls(s,&cfg);
		if( r != 0 ) {
			throw exception(r,"nng_http_server_get_tls");
		}
		return cfg;
	}
	
	void set_tls( tls::config_view cfg ) const {
		int r = nng_http_server_set_tls(s,cfg.get());
		if( r != 0 ) {
			throw exception(r,"nng_http_server_set_tls");
		}
	}
	
	void set_error_page( uint16_t code, const char* body ) const {
		int r = nng_http_server_set_error_page(s,code,body);
		if( r != 0 ) {
			throw exception(r,"nng_http_server_set_error_page");
		}
	}
	
	void set_error_file( uint16_t code, const char* path ) const {
		int r = nng_http_server_set_error_file(s,code,path);
		if( r != 0 ) {
			throw exception(r,"nng_http_server_set_error_file");
		}
	}
	
	void set_res_error( res_view res ) const {
		int r = nng_http_server_res_error(s,res.get());
		if( r != 0 ) {
			throw exception(r,"nng_http_server_res_error");
		}
	}
	
};

}}

#endif
