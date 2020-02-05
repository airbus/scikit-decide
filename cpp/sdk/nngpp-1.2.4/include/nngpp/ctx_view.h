#ifndef NNGPP_CTX_VIEW_H
#define NNGPP_CTX_VIEW_H
#include "aio_view.h"

namespace nng {

struct ctx_view {
protected:
	nng_ctx c{0};
	
public:
	ctx_view() = default;

	ctx_view( nng_ctx c ) noexcept : c(c) {}

	nng_ctx get() const noexcept {
		return c;
	}

	explicit operator bool() const noexcept {
		return c.id != 0;
	}
	
	int id() const noexcept {
		return nng_ctx_id(c);
	}
	
	void send( aio_view a ) const noexcept {
		nng_ctx_send(c,a.get());
	}

	void recv( aio_view a ) const noexcept {
		nng_ctx_recv(c,a.get());
	}
	
	void set_opt( const char* name, view v ) const {
		int r = nng_ctx_set(c,name,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set");
		}
	}
	
	void set_opt_bool( const char* name, bool value ) const {
		int r = nng_ctx_set_bool(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_bool");
		}
	}
	
	void set_opt_int( const char* name, int value ) const {
		int r = nng_ctx_set_int(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_int");
		}
	}
	
	void set_opt_size( const char* name, size_t value ) const {
		int r = nng_ctx_set_size(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_size");
		}
	}
	
	void set_opt_uint64( const char* name, uint64_t value ) const {
		int r = nng_ctx_set_uint64(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_uint64");
		}
	}
	
	void set_opt_string( const char* name, const char* value ) const {
		int r = nng_ctx_set_string(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_string");
		}
	}
	
	void set_opt_ptr( const char* name, void* value ) const {
		int r = nng_ctx_set_ptr(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_ptr");
		}
	}
	
	void set_opt_ms( const char* name, nng_duration value ) const {
		int r = nng_ctx_set_ms(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_ms");
		}
	}
	
	void set_opt_addr( const char* name, const nng_sockaddr* value ) const {
		int r = nng_ctx_set_addr(c,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_set_addr");
		}
	}
	
	size_t get_opt( const char* name, buffer& out ) const {
		size_t size = out.size();
		int r = nng_ctx_get(c,name,out.data(),&size);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get");
		}
		return size;
	}
	
	bool get_opt_bool( const char* name ) const {
		bool out;
		int r = nng_ctx_get_bool(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_bool");
		}
		return out;
	}
	
	int get_opt_int( const char* name ) const {
		int out;
		int r = nng_ctx_get_int(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_int");
		}
		return out;
	}
	
	size_t get_opt_size( const char* name ) const {
		size_t out;
		int r = nng_ctx_get_size(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_size");
		}
		return out;
	}
	
	uint64_t get_opt_uint64( const char* name ) const {
		uint64_t out;
		int r = nng_ctx_get_uint64(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_uint64");
		}
		return out;
	}
	
	buffer get_opt_string( const char* name ) const {
		char* out;
		int r = nng_ctx_get_string(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_string");
		}
		return buffer(out, strlen(out) + 1);
	}
	
	void* get_opt_ptr( const char* name ) const {
		void* out;
		int r = nng_ctx_get_ptr(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_ptr");
		}
		return out;
	}
	
	nng_duration get_opt_ms( const char* name ) const {
		nng_duration out;
		int r = nng_ctx_get_ms(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_ms");
		}
		return out;
	}
	
	nng_sockaddr get_opt_addr( const char* name ) const {
		nng_sockaddr out;
		int r = nng_ctx_get_addr(c,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_ctx_get_addr");
		}
		return out;
	}
	
};

inline nng_duration get_opt_recv_timeout( ctx_view s ) {
	return s.get_opt_ms( to_name(option::recv_timeout) );
}

inline nng_duration get_opt_send_timeout( ctx_view s ) {
	return s.get_opt_ms( to_name(option::send_timeout) );
}

inline void set_opt_recv_timeout( ctx_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::recv_timeout), v );
}

inline void set_opt_send_timeout( ctx_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::send_timeout), v );
}

}

#endif
