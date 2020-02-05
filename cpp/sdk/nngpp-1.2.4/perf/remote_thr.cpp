// This is a port of nng perf to nngpp
// See https://github.com/nanomsg/nng/tree/master/perf

#include <cstdio>
#include "throughput.h"
#include "misc.h"

int main(int argc, char** argv) try {
	if(argc != 4) {
		fprintf(stderr,"Usage: remote_thr <connect-to> <msg-size> <count>");
		return 1;
	}

	auto msgsize = parse_int(argv[2], "message size");
	auto trips   = parse_int(argv[3], "count");

	throughput_client(argv[1], msgsize, trips);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
