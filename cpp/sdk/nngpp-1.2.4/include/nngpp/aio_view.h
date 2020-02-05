#ifndef NNGPP_AIO_VIEW_H
#define NNGPP_AIO_VIEW_H
#include "msg.h"

namespace nng {

struct aio_view {
protected:
	nng_aio* a = nullptr;
	
public:
	aio_view() = default;

	aio_view( nng_aio* a ) noexcept : a(a) {}

	nng_aio* get() const noexcept {
		return a;
	}

	nng_aio* operator->() const noexcept {
		return a;
	}

	explicit operator bool() const noexcept {
		return a != nullptr;
	}
	
	void stop() const noexcept {
		nng_aio_stop(a);
	}
	
	error result() const noexcept {
		return (error)nng_aio_result(a);
	}
	
	size_t count() const noexcept {
		return nng_aio_count(a);
	}
	
	void cancel() const noexcept {
		nng_aio_cancel(a);
	}
	
	void abort( int err_code ) const noexcept {
		nng_aio_abort(a,err_code);
	}
	
	void wait() const noexcept {
		nng_aio_wait(a);
	}
	
	void set_msg( msg_view m ) const noexcept {
		nng_aio_set_msg(a,m.get());
	}

	void set_msg( msg&& m ) const noexcept {
		set_msg(m.release());
	}
	
	msg_view get_msg() const noexcept {
		return nng_aio_get_msg(a);
	}
	
	msg release_msg() const noexcept {
		auto m = nng_aio_get_msg(a);
		nng_aio_set_msg(a,nullptr);
		return msg(m);
	}

	template<typename T = void>
	void set_input( unsigned int index, T* arg ) const {
		int r = nng_aio_set_input(a,index,(void*)arg);
		if( r != 0 ) {
			throw exception(r,"nng_aio_set_input");
		}
	}

	template<typename T = void>
	T* get_input( unsigned int index ) const noexcept {
		return (T*)nng_aio_get_input(a,index);
	}
	
	template<typename T = void>
	void set_output( unsigned int index, T* arg ) const {
		int r = nng_aio_set_output(a,index,(void*)arg);
		if( r != 0 ) {
			throw exception(r,"nng_aio_set_output");
		}
	}
	
	template<typename T = void>
	T* get_output( unsigned int index ) const noexcept {
		return (T*)nng_aio_get_output(a,index);
	}
	
	void set_timeout( nng_duration when ) const noexcept {
		nng_aio_set_timeout(a,when);
	}
	
	void set_iov( unsigned int niov, const nng_iov* iov ) const {
		int r = nng_aio_set_iov(a,niov,iov);
		if( r != 0 ) {
			throw exception(r,"nng_aio_set_iov");
		}
	}
	
	void set_iov( const nng_iov& iov ) const {
		set_iov(1,&iov);
	}
	
	bool begin() const noexcept {
		return nng_aio_begin(a);
	}
	
	void finish( error e = error::success ) const noexcept {
		nng_aio_finish(a,(int)e);
	}
	
	void defer( nng_aio_cancelfn fn, void* arg ) const noexcept {
		nng_aio_defer(a,fn,arg);
	}
	
};

inline void sleep( nng_duration ms, aio_view a ) noexcept {
	nng_sleep_aio(ms,a.get());
}

}

#endif
