// This is a port of nng perf to nngpp
// See https://github.com/nanomsg/nng/tree/master/perf

#include <cstdio>
#include "latency.h"
#include "misc.h"

int main(int argc, char** argv) try {
	if(argc != 4) {
		fprintf(stderr,"Usage: local_lat <listen-addr> <msg-size> <roundtrips>");
		return 1;
	}

	auto msgsize = parse_int(argv[2], "Invalid message size");
	auto trips   = parse_int(argv[3], "Invalid round-trips");

	latency_server(argv[1], msgsize, trips);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
