#ifndef NNGPP_PUB0_H
#define NNGPP_PUB0_H
#include <nngpp/socket.h>
#include <nng/protocol/pubsub0/pub.h>

namespace nng { namespace pub {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_pub0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pub0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_pub0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pub0_open_raw");
	}
	return socket(s);
}

}
}}

#endif
