#ifndef NNGPP_PAIR0_H
#define NNGPP_PAIR0_H
#include <nngpp/socket.h>
#include <nng/protocol/pair0/pair.h>

namespace nng { namespace pair { namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_pair0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pair0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_pair0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_pair0_open_raw");
	}
	return socket(s);
}

}}}

#endif
