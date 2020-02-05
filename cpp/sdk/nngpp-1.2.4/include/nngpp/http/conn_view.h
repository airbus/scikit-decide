#ifndef NNGPP_HTTP_CONN_VIEW_H
#define NNGPP_HTTP_CONN_VIEW_H
#include <nngpp/aio_view.h>
#include "req_view.h"
#include "res_view.h"

namespace nng { namespace http {

struct conn_view {
protected:
	nng_http_conn* n = nullptr;
	
public:
	conn_view() = default;

	conn_view( nng_http_conn* n ) noexcept : n(n) {}
	
	nng_http_conn* get() const noexcept {
		return n;
	}

	nng_http_conn* operator->() const noexcept {
		return n;
	}

	explicit operator bool() const noexcept {
		return n != nullptr;
	}
	
	void read( aio_view a ) const noexcept {
		nng_http_conn_read(n,a.get());
	}
	
	void read_all( aio_view a ) const noexcept {
		nng_http_conn_read_all(n,a.get());
	}
	
	void read( req_view q, aio_view a ) const noexcept {
		nng_http_conn_read_req(n,q.get(),a.get());
	}
	
	void read( res_view s, aio_view a ) const noexcept {
		nng_http_conn_read_res(n,s.get(),a.get());
	}
	
	void write( aio_view a ) const noexcept {
		nng_http_conn_write(n,a.get());
	}
	
	void write_all( aio_view a ) const noexcept {
		nng_http_conn_write_all(n,a.get());
	}
	
	void write( req_view q, aio_view a ) const noexcept {
		nng_http_conn_write_req(n,q.get(),a.get());
	}
	
	void write( res_view s, aio_view a ) const noexcept {
		nng_http_conn_write_res(n,s.get(),a.get());
	}
	
	void hijack() const {
		int r = nng_http_hijack(n);
		if( r != 0 ) {
			throw exception(r,"nng_http_hijack");
		}
	}

	void transact( req_view q, res_view s, aio_view a ) const noexcept {
		nng_http_conn_transact(n,q.get(),s.get(),a.get());
	}
};

}}

#endif
