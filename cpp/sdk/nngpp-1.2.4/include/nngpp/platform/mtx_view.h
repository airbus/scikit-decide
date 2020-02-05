#ifndef NNGPP_MTX_VIEW_H
#define NNGPP_MTX_VIEW_H
#include <nng/nng.h>
#include <nng/supplemental/util/platform.h>

namespace nng {

struct mtx_view {
protected:
	nng_mtx* m = nullptr;
	
public:
	mtx_view() = default;
	
	mtx_view( nng_mtx* m ) noexcept : m(m) {}

	nng_mtx* get() const noexcept {
		return m;
	}

	nng_mtx* operator->() const noexcept {
		return m;
	}

	explicit operator bool() const noexcept {
		return m != nullptr;
	}

	void lock() const noexcept {
		nng_mtx_lock(m);
	}

	void unlock() const noexcept {
		nng_mtx_unlock(m);
	}

};

}

#endif
