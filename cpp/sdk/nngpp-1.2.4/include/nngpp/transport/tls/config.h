#ifndef NNGPP_TLS_CONFIG_H
#define NNGPP_TLS_CONFIG_H
#include "config_view.h"

namespace nng { namespace tls {

struct config : config_view {

	config() = default;
	
	explicit config( nng_tls_config* c ) noexcept : config_view(c) {}
	
	explicit config( mode mode ) {
		int r = nng_tls_config_alloc( &c, (nng_tls_mode)mode );
		if( r != 0 ) {
			throw exception(r,"nng_tls_config_alloc");
		}
	}
	
	config( const config& rhs ) = delete;
	
	config( config&& rhs ) noexcept : config_view(rhs.c) {
		rhs.c = nullptr;
	}
	
	config& operator=( const config& rhs ) = delete;
	
	config& operator=( config&& rhs ) noexcept {
		if( this != &rhs ) {
			if( c != nullptr ) nng_tls_config_free(c);
			c = rhs.c;
			rhs.c = nullptr;
		}
		return *this;
	}
	
	~config() {
		if( c != nullptr ) nng_tls_config_free(c);
	}
	
	nng_tls_config* release() noexcept {
		auto out = c;
		c = nullptr;
		return out;
	}

};

inline config config_view::hold() const noexcept {
	nng_tls_config_hold(c);
	return config(c);
}

inline config make_config( mode mode ) {
	return config(mode);
}

}}

#endif
