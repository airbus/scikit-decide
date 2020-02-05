#ifndef NNGPP_PUSHE0_H
#define NNGPP_PUSHE0_H
#include <nngpp/socket.h>
#include <nng/protocol/pipeline0/push.h>

namespace nng { namespace push {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_push0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_push0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_push0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_push0_open_raw");
	}
	return socket(s);
}

}
}}

#endif
