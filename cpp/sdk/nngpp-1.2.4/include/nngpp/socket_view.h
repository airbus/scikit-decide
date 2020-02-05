#ifndef NNGPP_SOCKET_VIEW_H
#define NNGPP_SOCKET_VIEW_H
#include "core.h"
#include "option.h"
#include "aio_view.h"

namespace nng {

struct socket_view {
protected:
	nng_socket s{0};
	
public:

	socket_view() = default;

	socket_view( nng_socket s ) noexcept : s(s) {}
	
	nng_socket get() const noexcept {
		return s;
	}
	
	explicit operator bool() const noexcept {
		return s.id != 0;
	}
	
	int id() const noexcept {
		return nng_socket_id(s);
	}

	void dial( const char* addr, int flags = 0 ) const {
		int r = nng_dial( s, addr, nullptr, flags );
		if( r != 0 ) {
			throw exception(r,"nng_dial");
		}
	}
	
	void listen( const char* addr, int flags = 0 ) const {
		int r = nng_listen( s, addr, nullptr, flags );
		if( r != 0 ) {
			throw exception(r,"nng_listen");
		}
	}

	void send( view buf, int flags = 0 ) const {
		int r = nng_send(s,buf.data(),buf.size(),flags & ~flag::alloc);
		if( r != 0 ) {
			throw exception(r,"nng_send");
		}
	}

	void send( buffer&& buf, int flags = 0 ) const {
		int r = nng_send(s,buf.data(),buf.size(),flags | flag::alloc);
		if( r != 0 ) {
			throw exception(r,"nng_send");
		}
		// if successful, the buffer has been freed
		buf.release();
	}

	void send( msg&& m, int flags = 0 ) const {
		int r = nng_sendmsg(s,m.get(),flags);
		if( r != 0 ) {
			throw exception(r,"nng_sendmsg");
		}
		// if successful, the msg is owned by the socket
		m.release();
	}

	/**
	 * Send asynchronously
	 */
	void send( aio_view a ) const noexcept {
		nng_send_aio(s,a.get());
	}

	/**
	 * Receives into a pre-allocated buffer
	 * Synchronous
	 * Returns the number of bytes written
	 */
	size_t recv( view out, int flags = 0 ) const {
		size_t sz = out.size();
		int r = nng_recv(s,out.data(),&sz,flags & ~flag::alloc);
		if( r != 0 ) {
			throw exception(r,"nng_recv");
		}
		return sz;
	}

	/**
	 * Receives into a buffer
	 * Synchronous
	 * Allocates and copies
	 */
	buffer recv( int flags = 0 ) const {
		void* data;
		size_t size;
		int r = nng_recv(s,&data,&size,flags | flag::alloc);
		if( r != 0 ) {
			throw exception(r,"nng_recv");
		}
		return buffer(data,size);
	}

	/**
	 * Receive a msg object
	 * Synchronous
	 * No allocation
	 * No copy
	 */
	msg recv_msg( int flags = 0 ) const {
		nng_msg* m;
		int r = nng_recvmsg(s,&m,flags);
		if( r != 0 ) {
			throw exception(r,"nng_recvmsg");
		}
		return msg(m);
	}

	/**
	 * Receive asynchronously
	 */
	void recv( aio_view a ) const noexcept {
		nng_recv_aio(s,a.get());
	}
	
	void pipe_notify( pipe_ev ev, nng_pipe_cb cb, void* arg ) const {
		int r = nng_pipe_notify( s, (nng_pipe_ev)(int)ev, cb, arg );
		if( r != 0 ) {
			throw exception(r,"nng_pipe_notify");
		}
	}
	
	void set_opt( const char* name, view v ) const {
		int r = nng_socket_set(s,name,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_socket_set");
		}
	}
	
	void set_opt_bool( const char* name, bool value ) const {
		int r = nng_socket_set_bool(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_bool");
		}
	}
	
	void set_opt_int( const char* name, int value ) const {
		int r = nng_socket_set_int(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_int");
		}
	}
	
	void set_opt_size( const char* name, size_t value ) const {
		int r = nng_socket_set_size(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_size");
		}
	}
	
	void set_opt_uint64( const char* name, uint64_t value ) const {
		int r = nng_socket_set_uint64(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_uint64");
		}
	}
	
	void set_opt_string( const char* name, const char* value ) const {
		int r = nng_socket_set_string(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_string");
		}
	}
	
	void set_opt_ptr( const char* name, void* value ) const {
		int r = nng_socket_set_ptr(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_ptr");
		}
	}
	
	void set_opt_ms( const char* name, nng_duration value ) const {
		int r = nng_socket_set_ms(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_ms");
		}
	}
	
	void set_opt_addr( const char* name, const nng_sockaddr* value ) const {
		int r = nng_socket_set_addr(s,name,value);
		if( r != 0 ) {
			throw exception(r,"nng_socket_set_addr");
		}
	}
	
	size_t get_opt( const char* name, buffer& out ) const {
		size_t size = out.size();
		int r = nng_socket_get(s,name,out.data(),&size);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get");
		}
		return size;
	}
	
	bool get_opt_bool( const char* name ) const {
		bool out;
		int r = nng_socket_get_bool(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_bool");
		}
		return out;
	}
	
	int get_opt_int( const char* name ) const {
		int out;
		int r = nng_socket_get_int(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_int");
		}
		return out;
	}
	
	size_t get_opt_size( const char* name ) const {
		size_t out;
		int r = nng_socket_get_size(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_size");
		}
		return out;
	}
	
	uint64_t get_opt_uint64( const char* name ) const {
		uint64_t out;
		int r = nng_socket_get_uint64(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_uint64");
		}
		return out;
	}
	
	buffer get_opt_string( const char* name ) const {
		char* data;
		int r = nng_socket_get_string(s,name,&data);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_string");
		}
		return buffer(data,strlen(data)+1);
	}

	void* get_opt_ptr( const char* name ) const {
		void* out;
		int r = nng_socket_get_ptr(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_ptr");
		}
		return out;
	}
	
	nng_duration get_opt_ms( const char* name ) const {
		nng_duration out;
		int r = nng_socket_get_ms(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_ms");
		}
		return out;
	}

	nng_sockaddr get_opt_addr( const char* name ) const {
		nng_sockaddr out;
		int r = nng_socket_get_addr(s,name,&out);
		if( r != 0 ) {
			throw exception(r,"nng_socket_get_addr");
		}
		return out;
	}
	
};

inline socket_view pipe_view::get_socket() const noexcept {
	return nng_pipe_socket(p);
}

inline void device( socket_view lhs, socket_view rhs ) {
	int r = nng_device(lhs.get(),rhs.get());
	if( r != 0 ) {
		throw exception(r,"nng_device");
	}
}

inline buffer get_opt_socket_name( socket_view s ) {
	return s.get_opt_string( to_name(option::socket_name) );
}

inline bool get_opt_raw( socket_view s ) {
	return s.get_opt_bool( to_name(option::raw) );
}

inline int get_opt_protocol( socket_view s ) {
	return s.get_opt_int( to_name(option::protocol) );
}

inline buffer get_opt_protocol_name( socket_view s ) {
	return s.get_opt_string( to_name(option::protocol_name) );
}

inline int get_opt_peer( socket_view s ) {
	return s.get_opt_int( to_name(option::peer) );
}

inline buffer get_opt_peer_name( socket_view s ) {
	return s.get_opt_string( to_name(option::peer_name) );
}

inline int get_opt_recv_buffer( socket_view s ) {
	return s.get_opt_int( to_name(option::recv_buffer) );
}

inline int get_opt_send_buffer( socket_view s ) {
	return s.get_opt_int( to_name(option::send_buffer) );
}

inline int get_opt_recv_fd( socket_view s ) {
	return s.get_opt_int( to_name(option::recv_fd) );
}

inline int get_opt_send_fd( socket_view s ) {
	return s.get_opt_int( to_name(option::send_fd) );
}

inline nng_duration get_opt_recv_timeout( socket_view s ) {
	return s.get_opt_ms( to_name(option::recv_timeout) );
}

inline nng_duration get_opt_send_timeout( socket_view s ) {
	return s.get_opt_ms( to_name(option::send_timeout) );
}

inline int get_opt_ttl_max( socket_view s ) {
	return s.get_opt_int( to_name(option::ttl_max) );
}

inline buffer get_opt_recv_size_max( socket_view s ) {
	return s.get_opt_string( to_name(option::recv_size_max) );
}

inline nng_duration get_opt_reconnect_time_min( socket_view s ) {
	return s.get_opt_ms( to_name(option::reconnect_time_min) );
}

inline nng_duration get_opt_reconnect_time_max( socket_view s ) {
	return s.get_opt_ms( to_name(option::reconnect_time_max) );
}

inline void set_opt_socket_name( socket_view s, const char* v ) {
	s.set_opt_string( to_name(option::socket_name), v );
}

inline void set_opt_recv_buffer( socket_view s, int v ) {
	s.set_opt_int( to_name(option::recv_buffer), v );
}

inline void set_opt_send_buffer( socket_view s, int v ) {
	s.set_opt_int( to_name(option::send_buffer), v );
}

inline void set_opt_recv_timeout( socket_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::recv_timeout), v );
}

inline void set_opt_send_timeout( socket_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::send_timeout), v );
}

inline void set_opt_ttl_max( socket_view s, int v ) {
	s.set_opt_int( to_name(option::ttl_max), v );
}
inline void set_opt_reconnect_time_min( socket_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::reconnect_time_min), v );
}

inline void set_opt_reconnect_time_max( socket_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::reconnect_time_max), v );
}

}

#endif
