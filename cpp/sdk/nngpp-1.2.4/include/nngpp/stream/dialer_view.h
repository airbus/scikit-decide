#ifndef NNGPP_STREAM_DIALER_VIEW_H
#define NNGPP_STREAM_DIALER_VIEW_H
#include <nngpp/aio_view.h>

namespace nng { namespace stream {

struct dialer_view {
protected:
	nng_stream_dialer* d = nullptr;
	
public:
	dialer_view() = default;
	
	dialer_view( nng_stream_dialer* d ) noexcept : d(d) {}

	nng_stream_dialer* get() const noexcept {
		return d;
	}

	nng_stream_dialer* operator->() const noexcept {
		return d;
	}

	explicit operator bool() const noexcept {
		return d != nullptr;
	}
	
	void close() const noexcept {
		nng_stream_dialer_close(d);
	}
	
	void dial( aio_view a ) const noexcept {
		nng_stream_dialer_dial(d,a.get());
	}
	
	void get( const char* name, void* val, size_t* szp ) const {
		int r = nng_stream_dialer_get(d,name,val,szp);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get");
		}
	}
	
	template<typename T>
	T get( const char* name ) const {
		T out;
		size_t size = sizeof(out);
		get(name,&out,&size);
		return out;
	}

	bool get_bool( const char* name ) const {
		bool out;
		int r = nng_stream_dialer_get_bool(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_bool");
		}
		return out;
	}
	
	int get_int( const char* name ) const {
		int out;
		int r = nng_stream_dialer_get_int(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_int");
		}
		return out;
	}
	
	size_t get_size( const char* name ) const {
		size_t out;
		int r = nng_stream_dialer_get_size(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_size");
		}
		return out;
	}
	
	uint64_t get_uint64( const char* name ) const {
		uint64_t out;
		int r = nng_stream_dialer_get_uint64(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_uint64");
		}
		return out;
	}

	buffer get_string( const char* name ) const {
		char* data;
		int r = nng_stream_dialer_get_string(d,name,&data);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_string");
		}
		return buffer(data,strlen(data)+1);
	}

	void* get_ptr( const char* name ) const {
		void* out;
		int r = nng_stream_dialer_get_ptr(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_ptr");
		}
		return out;
	}
	
	nng_duration get_ms( const char* name ) const {
		nng_duration out;
		int r = nng_stream_dialer_get_ms(d,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_ms");
		}
		return out;
	}
	
	nng_sockaddr get_addr( const char* name ) const {
		nng_sockaddr data;
		int r = nng_stream_dialer_get_addr(d,name,&data);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_get_addr");
		}
		return data;
	}
	
	void set( const char* name, const void* val, size_t sz ) const {
		int r = nng_stream_dialer_set(d,name,val,sz);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set");
		}
	}
	
	template<typename T>
	void set( const char* name, const T& v ) const {
		set(name,&v,sizeof(v));
	}

	void set_bool( const char* name, bool value ) const {
		int r = nng_stream_dialer_set_bool(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_bool");
		}
	}
	
	void set_int( const char* name, int value ) const {
		int r = nng_stream_dialer_set_int(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_int");
		}
	}
	
	void set_size( const char* name, size_t value ) const {
		int r = nng_stream_dialer_set_size(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_size");
		}
	}
	
	void set_uint64( const char* name, uint64_t value ) const {
		int r = nng_stream_dialer_set_uint64(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_uint64");
		}
	}
	
	void set_string( const char* name, const char* value ) const {
		int r = nng_stream_dialer_set_string(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_string");
		}
	}
	
	void set_ptr( const char* name, void* value ) const {
		int r = nng_stream_dialer_set_ptr(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_ptr");
		}
	}
	
	void set_ms( const char* name, nng_duration value ) const {
		int r = nng_stream_dialer_set_ms(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_ms");
		}
	}
	
	void set_addr( const char* name, const nng_sockaddr* value ) const {
		int r = nng_stream_dialer_set_addr(d,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_dialer_set_addr");
		}
	}
	
};

}}

#endif
