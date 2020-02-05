#ifndef NNGPP_REP0_H
#define NNGPP_REP0_H
#include <nngpp/socket.h>
#include <nng/protocol/reqrep0/rep.h>

namespace nng { namespace rep {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_rep0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_rep0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_rep0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_rep0_open_raw");
	}
	return socket(s);
}

}
}}

#endif
