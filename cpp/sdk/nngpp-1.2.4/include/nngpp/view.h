#ifndef NNGPP_VIEW_H
#define NNGPP_VIEW_H
#include <cstring>
#include "error.h"

namespace nng {

struct view {
protected:
	using pointer = void*;
	pointer d = nullptr;
	size_t s = 0;
	
public:
	view() = default;
	
	template<typename T>
	view( T* d, size_t s ) noexcept : d((pointer)d), s(s) {}

	template<typename T,size_t N>
	view( T(&d)[N] ) noexcept : d((pointer)d), s(N * sizeof(T)) {}

	view( nng_iov x ) noexcept : view(x.iov_buf,x.iov_len) {}
	
	template<typename T = void>
	T* data() const noexcept {
		return (T*)d;
	}
	
	size_t size() const noexcept {
		return s;
	}

	explicit operator bool() const noexcept {
		return d != nullptr;
	}

	bool operator==( const view& rhs ) const noexcept {
		return s == rhs.s && memcmp(d,rhs.d,s) == 0;
	}

	bool operator!=( const view& rhs ) const noexcept {
		return !(*this == rhs);
	}

	operator nng_iov() const noexcept {
		return { d, s };
	}

};

}

#endif
