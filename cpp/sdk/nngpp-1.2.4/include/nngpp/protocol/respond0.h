#ifndef NNGPP_RESPOND0_H
#define NNGPP_RESPOND0_H
#include <nngpp/socket.h>
#include <nng/protocol/survey0/respond.h>

namespace nng { namespace respond {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_respondent0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_respondent0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_respondent0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_respondent0_open_raw");
	}
	return socket(s);
}

}
}}

#endif
