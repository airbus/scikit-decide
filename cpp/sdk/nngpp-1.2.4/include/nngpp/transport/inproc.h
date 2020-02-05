#ifndef NNGPP_TRANSPORT_INPROC_H
#define NNGPP_TRANSPORT_INPROC_H
#include <nngpp/error.h>
#include <nng/transport/inproc/inproc.h>

namespace nng { namespace inproc {

inline void register_transport() {
	int r = nng_inproc_register();
	if( r != 0 ) {
		throw exception(r,"nng_inproc_register");
	}
}

}}

#endif
