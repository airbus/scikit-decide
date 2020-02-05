#ifndef NNGPP_HTTP_CLIENT_VIEW_H
#define NNGPP_HTTP_CLIENT_VIEW_H
#include <nngpp/aio_view.h>
#include <nngpp/transport/tls/config_view.h>
#include "req_view.h"
#include "res_view.h"

namespace nng { namespace http {

struct client_view {
protected:
	nng_http_client* c = nullptr;
	
public:
	client_view() = default;
	
	client_view( nng_http_client* c ) noexcept : c(c) {}

	nng_http_client* get() const noexcept {
		return c;
	}

	nng_http_client* operator->() const noexcept {
		return c;
	}

	explicit operator bool() const noexcept {
		return c != nullptr;
	}
	
	tls::config_view get_tls() const {
		nng_tls_config* cfg;
		int r = nng_http_client_get_tls(c,&cfg);
		if( r != 0 ) {
			throw exception(r,"nng_http_client_get_tls");
		}
		return cfg;
	}

	void set_tls( tls::config_view cfg ) const {
		int r = nng_http_client_set_tls(c,cfg.get());
		if( r != 0 ) {
			throw exception(r,"nng_http_client_set_tls");
		}
	}
	
	void connect( aio_view a ) const noexcept {
		nng_http_client_connect(c,a.get());
	}
	
	void transact( req_view q, res_view s, aio_view a ) const noexcept {
		nng_http_client_transact(c,q.get(),s.get(),a.get());
	}
};

}}

#endif
