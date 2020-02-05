// This is a port of nng perf to nngpp
// See https://github.com/nanomsg/nng/tree/master/perf

#include <cstdio>
#include "latency.h"
#include "inproc.h"

int main(int argc, char** argv) try {
	if(argc != 3) {
		fprintf(stderr,"Usage: inproc_lat <msg-size> <count>");
		return 1;
	}

	inproc_args ia;
	ia.addr    = "inproc://latency_test";
	ia.msgsize = parse_int(argv[1], "Invalid message size");
	ia.count   = parse_int(argv[2], "Invalid count");
	ia.func    = latency_server;

	nng::thread thr( do_inproc, &ia );

	// Sleep a bit.
	nng::msleep(100);

	latency_client("inproc://latency_test", ia.msgsize, ia.count);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
