#ifndef NNGPP_BUS0_H
#define NNGPP_BUS0_H
#include <nngpp/socket.h>
#include <nng/protocol/bus0/bus.h>

namespace nng { namespace bus {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_bus0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_bus0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_bus0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_bus0_open_raw");
	}
	return socket(s);
}

}
}}

#endif
