#ifndef NNGPP_THREAD_VIEW_H
#define NNGPP_THREAD_VIEW_H
#include <nng/nng.h>
#include <nng/supplemental/util/platform.h>

namespace nng {

struct thread_view {
protected:
	nng_thread* t = nullptr;
	
public:
	thread_view() = default;
	
	thread_view( nng_thread* t ) noexcept : t(t) {}

	nng_thread* get() const noexcept {
		return t;
	}

	nng_thread* operator->() const noexcept {
		return t;
	}

	explicit operator bool() const noexcept {
		return t != nullptr;
	}

};

}

#endif
