#ifndef NNGPP_URL_VIEW_H
#define NNGPP_URL_VIEW_H
#include "view.h"

namespace nng {

struct url_view {
protected:
	nng_url* u = nullptr;
	
public:
	url_view() = default;
	
	url_view( nng_url* u ) noexcept : u(u) {}
	
	nng_url* get() const noexcept {
		return u;
	}
	
	nng_url* operator->() const noexcept {
		return u;
	}

	explicit operator bool() const noexcept {
		return u != nullptr;
	}

	bool operator==( const url_view& rhs ) const noexcept {
		view view_left(u->u_rawurl,strlen(u->u_rawurl));
		view view_right(rhs.u->u_rawurl,strlen(rhs.u->u_rawurl));
		return view_left == view_right;
	}

	bool operator!=( const url_view& rhs ) const noexcept {
		return !(*this == rhs);
	}

};

}

#endif
