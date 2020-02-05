#ifndef NNGPP_STREAM_STREAM_VIEW_H
#define NNGPP_STREAM_STREAM_VIEW_H
#include <nngpp/aio_view.h>

namespace nng { namespace stream {

struct stream_view {
protected:
	nng_stream* s = nullptr;
	
public:
	stream_view() = default;
	
	stream_view( nng_stream* s ) noexcept : s(s) {}

	nng_stream* get() const noexcept {
		return s;
	}

	nng_stream* operator->() const noexcept {
		return s;
	}

	explicit operator bool() const noexcept {
		return s != nullptr;
	}

	void close() const noexcept {
		nng_stream_close(s);
	}
	
	void send( nng::aio_view a ) const noexcept {
		nng_stream_send(s,a.get());
	}
	
	void recv( nng::aio_view a ) const noexcept {
		nng_stream_recv(s,a.get());
	}

	void get( const char* name, void* val, size_t* szp ) const {
		int r = nng_stream_get(s,name,val,szp);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get");
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
		int r = nng_stream_get_bool(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_bool");
		}
		return out;
	}
	
	int get_int( const char* name ) const {
		int out;
		int r = nng_stream_get_int(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_int");
		}
		return out;
	}
	
	size_t get_size( const char* name ) const {
		size_t out;
		int r = nng_stream_get_size(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_size");
		}
		return out;
	}
	
	uint64_t get_uint64( const char* name ) const {
		uint64_t out;
		int r = nng_stream_get_uint64(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_uint64");
		}
		return out;
	}

	buffer get_string( const char* name ) const {
		char* data;
		int r = nng_stream_get_string(s,name,&data);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_string");
		}
		return buffer(data,strlen(data)+1);
	}

	void* get_ptr( const char* name ) const {
		void* out;
		int r = nng_stream_get_ptr(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_ptr");
		}
		return out;
	}
	
	nng_duration get_ms( const char* name ) const {
		nng_duration out;
		int r = nng_stream_get_ms(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_ms");
		}
		return out;
	}
	
	nng_sockaddr get_addr( const char* name ) const {
		nng_sockaddr data;
		int r = nng_stream_get_addr(s,name,&data);
		if( r != 0 ) {
			throw exception(r,"nng_stream_get_addr");
		}
		return data;
	}
	
	void set( const char* name, const void* val, size_t sz ) const {
		int r = nng_stream_set(s,name,val,sz);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set");
		}
	}
	
	template<typename T>
	void set( const char* name, const T& v ) const {
		set(name,&v,sizeof(v));
	}

	void set_bool( const char* name, bool value ) const {
		int r = nng_stream_set_bool(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_bool");
		}
	}
	
	void set_int( const char* name, int value ) const {
		int r = nng_stream_set_int(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_int");
		}
	}
	
	void set_size( const char* name, size_t value ) const {
		int r = nng_stream_set_size(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_size");
		}
	}
	
	void set_uint64( const char* name, uint64_t value ) const {
		int r = nng_stream_set_uint64(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_uint64");
		}
	}
	
	void set_string( const char* name, const char* value ) const {
		int r = nng_stream_set_string(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_string");
		}
	}
	
	void set_ptr( const char* name, void* value ) const {
		int r = nng_stream_set_ptr(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_ptr");
		}
	}
	
	void set_ms( const char* name, nng_duration value ) const {
		int r = nng_stream_set_ms(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_ms");
		}
	}
	
	void set_addr( const char* name, const nng_sockaddr& value ) const {
		int r = nng_stream_set_addr(s,name,&value);
		if( r != 0 ) {
			throw exception(r,"nng_stream_set_addr");
		}
	}

};

}}

#endif
