#ifndef NNGPP_PULL0_H
#define NNGPP_PULL0_H
#include <nngpp/socket.h>
#include <nng/protocol/pipeline0/pull.h>

namespace nng { namespace pull {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_pull0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pull0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_pull0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pull0_open_raw");
	}
	return socket(s);
}

}
}}

#endif
