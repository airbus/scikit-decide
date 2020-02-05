#ifndef NNGPP_STAT_H
#define NNGPP_STAT_H
#include "stat_view.h"
#include "error.h"

namespace nng {

struct stat : stat_view {

	stat() = default;

	explicit stat( nng_stat* s ) noexcept : stat_view(s) {}

	stat( const stat& rhs ) = delete;
	
	stat( stat&& rhs ) noexcept : stat_view(rhs.s) {
		rhs.s = nullptr;
	}
	
	stat& operator=( const stat& rhs ) = delete;
	
	stat& operator=( stat&& rhs ) noexcept {
		if( this != &rhs ) {
			if( s != nullptr ) nng_stats_free(s);
			s = rhs.s;
			rhs.s = nullptr;
		}
		return *this;
	}
	
	~stat() {
		if( s != nullptr ) nng_stats_free(s);
	}

	nng_stat* release() noexcept {
		auto out = s;
		s = nullptr;
		return out;
	}

};

inline stat make_stat() {
	nng_stat* s;
	int r = nng_stats_get(&s);
	if( r != 0 ) {
		throw exception(r,"nng_stats_get");
	}
	return stat(s);
}

}

#endif
