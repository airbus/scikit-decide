#ifndef NNGPP_CV_VIEW_H
#define NNGPP_CV_VIEW_H
#include <nngpp/error.h>
#include <nng/supplemental/util/platform.h>

namespace nng {

struct cv_view {
protected:
	nng_cv* c = nullptr;
	
public:
	cv_view() = default;
	
	cv_view( nng_cv* c ) noexcept : c(c) {}

	nng_cv* get() const noexcept {
		return c;
	}

	nng_cv* operator->() const noexcept {
		return c;
	}

	explicit operator bool() const noexcept {
		return c != nullptr;
	}

	void wait() const noexcept {
		nng_cv_wait(c);
	}

	/** wait until time t
	 *  returns true if signalled, false if timed out
	 */
	bool wait_until( nng_time t ) const {
		int r = nng_cv_until(c,t);
		if( r != 0 && r != (int)error::timedout ) {
			throw exception(r,"nng_cv_until");
		}
		return r == 0;
	}

	void wake_all() const noexcept {
		nng_cv_wake(c);
	}

	void wake_one() const noexcept {
		nng_cv_wake1(c);
	}

};

}

#endif
