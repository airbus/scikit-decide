// This is a port of the nng demo to nngpp
// See https://github.com/nanomsg/nng/tree/master/demo/async

#include <cstdio>
#include <cstdlib>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/req0.h>
#include <nngpp/platform/platform.h>

//  The client runs just once, and then returns.
void client(const char* url, const char* msecstr) {
	auto msec = atoi(msecstr);

	auto sock = nng::req::open();
	sock.dial(url);

	auto start = nng::clock();
	auto msg = nng::make_msg(0);
	msg.body().append_u32(msec);
	sock.send(std::move(msg));
	msg = sock.recv_msg();
	auto end = nng::clock();

	printf("Request took %u milliseconds.\n", (uint32_t)(end - start));
}

int main(int argc, char** argv) try {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <url> <secs>\n", argv[0]);
		return 1;
	}
	client(argv[1], argv[2]);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
