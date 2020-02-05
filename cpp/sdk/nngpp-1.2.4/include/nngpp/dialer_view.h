#ifndef NNGPP_DIALER_VIEW_H
#define NNGPP_DIALER_VIEW_H
#include "option.h"
#include "buffer.h"

namespace nng {

struct dialer_view {
protected:
	nng_dialer d{0};
	
public:

	dialer_view() = default;
	
	dialer_view( nng_dialer d ) noexcept : d(d) {}
	
	nng_dialer get() const noexcept {
		return d;
	}

	explicit operator bool() const noexcept {
		return d.id != 0;
	}
	
	void start( int flags = 0 ) const {
		int r = nng_dialer_start(d,flags);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_start");
		}
	}
	
	int id() const noexcept {
		return nng_dialer_id(d);
	}
	
	void set_opt( const char* name, view v ) const {
		int r = nng_dialer_set(d,name,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set");
		}
	}
	
	void set_opt_bool( const char* name, bool val ) const {
		int r = nng_dialer_set_bool(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_bool");
		}
	}
	
	void set_opt_int( const char* name, int val ) const {
		int r = nng_dialer_set_int(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_int");
		}
	}
	
	void set_opt_size( const char* name, size_t val ) const {
		int r = nng_dialer_set_size(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_size");
		}
	}
	
	void set_opt_uint64( const char* name, uint64_t val ) const {
		int r = nng_dialer_set_uint64(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_uint64");
		}
	}
	
	void set_opt_string( const char* name, const char* val ) const {
		int r = nng_dialer_set_string(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_string");
		}
	}
	
	void set_opt_ptr( const char* name, void* val ) const {
		int r = nng_dialer_set_ptr(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_ptr");
		}
	}
	
	void set_opt_ms( const char* name, nng_duration val ) const {
		int r = nng_dialer_set_ms(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_ms");
		}
	}
	
	void set_opt_addr( const char* name, const nng_sockaddr* val ) const {
		int r = nng_dialer_set_addr(d,name,val);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_set_addr");
		}
	}

	size_t get_opt( const char* name, buffer& out ) const {
		size_t size = out.size();
		int r = nng_dialer_get(d,name,out.data(),&size);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get");
		}
		return size;
	}
	
	bool get_opt_bool( const char* name ) const {
		bool out;
		int r = nng_dialer_get_bool(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_bool");
		}
		return out;
	}
	
	int get_opt_int( const char* name ) const {
		int out;
		int r = nng_dialer_get_int(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_int");
		}
		return out;
	}
	
	size_t get_opt_size( const char* name ) const {
		size_t out;
		int r = nng_dialer_get_size(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_size");
		}
		return out;
	}
	
	uint64_t get_opt_uint64( const char* name ) const {
		uint64_t out;
		int r = nng_dialer_get_uint64(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_uint64");
		}
		return out;
	}
	
	buffer get_opt_string( const char* name ) const {
		char* data;
		int r = nng_dialer_get_string(d,name,&data);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_string");
		}
		return buffer(data,strlen(data)+1);
	}
	
	void* get_opt_ptr( const char* name ) const {
		void* out;
		int r = nng_dialer_get_ptr(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_ptr");
		}
		return out;
	}
	
	nng_duration get_opt_ms( const char* name ) const {
		nng_duration out;
		int r = nng_dialer_get_ms(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_ms");
		}
		return out;
	}
	
	nng_sockaddr get_opt_addr( const char* name ) const {
		nng_sockaddr out;
		int r = nng_dialer_get_addr(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_dialer_get_addr");
		}
		return out;
	}
	
};

inline buffer get_opt_url( dialer_view s ) {
	return s.get_opt_string( to_name(option::url) );
}

inline size_t get_opt_recv_size_max( dialer_view s ) {
	return s.get_opt_size( to_name(option::recv_size_max) );
}

inline nng_duration get_opt_reconnect_time_min( dialer_view s ) {
	return s.get_opt_ms( to_name(option::reconnect_time_min) );
}

inline nng_duration get_opt_reconnect_time_max( dialer_view s ) {
	return s.get_opt_ms( to_name(option::reconnect_time_max) );
}

inline void set_opt_recv_size_max( dialer_view s, size_t v ) {
	s.set_opt_size( to_name(option::recv_size_max), v );
}

inline void set_opt_reconnect_time_min( dialer_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::reconnect_time_min), v );
}

inline void set_opt_reconnect_time_max( dialer_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::reconnect_time_max), v );
}

}

#endif
