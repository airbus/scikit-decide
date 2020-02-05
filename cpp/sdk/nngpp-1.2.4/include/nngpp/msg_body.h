#ifndef NNGPP_MSG_BODY_H
#define NNGPP_MSG_BODY_H
#include "view.h"

namespace nng {

class msg_body {
	nng_msg* m;
	
public:
	
	explicit msg_body( nng_msg* m ) noexcept : m(m) {}

	nng_msg* get_msg() const noexcept {
		return m;
	}
	
	template<typename T = void>
	T* data() const noexcept {
		return (T*)nng_msg_body(m);
	}
	
	size_t size() const noexcept {
		return nng_msg_len(m);
	}

	view get() const noexcept {
		return view(
			nng_msg_body(m),
			nng_msg_len(m)
		);
	}
	
	void clear() const noexcept {
		nng_msg_clear(m);
	}
	
	void append( view v ) const {
		int r = nng_msg_append(m,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_msg_append");
		}
	}

	void insert( view v ) const {
		int r = nng_msg_insert(m,v.data(),v.size());
		if( r != 0 ) {
			throw exception(r,"nng_msg_insert");
		}
	}
	
	void trim( size_t size ) const {
		int r = nng_msg_trim(m,size);
		if( r != 0 ) {
			throw exception(r,"nng_msg_trim");
		}
	}
	
	void chop( size_t size ) const {
		int r = nng_msg_chop(m,size);
		if( r != 0 ) {
			throw exception(r,"nng_msg_chop");
		}
	}
	
	void append_u16( uint16_t val ) const {
		int r = nng_msg_append_u16(m,val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_append_u16");
		}
	}
	
	void append_u32( uint32_t val ) const {
		int r = nng_msg_append_u32(m,val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_append_u32");
		}
	}
	
	void append_u64( uint64_t val ) const {
		int r = nng_msg_append_u64(m,val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_append_u64");
		}
	}
	
	void insert_u16( uint16_t val ) const {
		int r = nng_msg_insert_u16(m,val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_insert_u16");
		}
	}
	
	void insert_u32( uint32_t val ) const {
		int r = nng_msg_insert_u32(m,val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_insert_u32");
		}
	}
	
	void insert_u64( uint64_t val ) const {
		int r = nng_msg_insert_u64(m,val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_insert_u64");
		}
	}
	
	uint16_t trim_u16() const {
		uint16_t val;
		int r = nng_msg_trim_u16(m,&val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_trim_u16");
		}
		return val;
	}
	
	uint32_t trim_u32() const {
		uint32_t val;
		int r = nng_msg_trim_u32(m,&val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_trim_u32");
		}
		return val;
	}
	
	uint64_t trim_u64() const {
		uint64_t val;
		int r = nng_msg_trim_u64(m,&val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_trim_u64");
		}
		return val;
	}
	
	uint16_t chop_u16() const {
		uint16_t val;
		int r = nng_msg_chop_u16(m,&val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_chop_u16");
		}
		return val;
	}
	
	uint32_t chop_u32() const {
		uint32_t val;
		int r = nng_msg_chop_u32(m,&val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_chop_u32");
		}
		return val;
	}
	
	uint64_t chop_u64() const {
		uint64_t val;
		int r = nng_msg_chop_u64(m,&val);
		if( r != 0 ) {
			throw exception(r,"nng_msg_chop_u64");
		}
		return val;
	}

};

}

#endif
