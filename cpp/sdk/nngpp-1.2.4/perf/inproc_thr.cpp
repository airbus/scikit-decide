// This is a port of nng perf to nngpp
// See https://github.com/nanomsg/nng/tree/master/perf

#include <cstdio>
#include "throughput.h"
#include "inproc.h"
#include "misc.h"

int main(int argc, char** argv) try {
	if(argc != 3) {
		fprintf(stderr,"Usage: inproc_thr <msg-size> <count>");
		return 1;
	}

	inproc_args ia;
	ia.addr    = "inproc://tput_test";
	ia.msgsize = parse_int(argv[1], "Invalid message size");
	ia.count   = parse_int(argv[2], "Invalid count");
	ia.func    = throughput_server;

	nng::thread thr( do_inproc, &ia );

	// Sleep a bit.
	nng::msleep(100);

	throughput_client("inproc://tput_test", ia.msgsize, ia.count);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
