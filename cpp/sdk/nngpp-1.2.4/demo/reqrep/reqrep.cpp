// This is a port of the nng demo to nngpp
// See https://github.com/nanomsg/nng/tree/master/demo/reqrep

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/req0.h>
#include <nngpp/protocol/rep0.h>

#define CLIENT "client"
#define SERVER "server"
#define DATECMD 1

#define PUT64(ptr, u) \
	do { \
		(ptr)[0] = (uint8_t)(((uint64_t)(u)) >> 56); \
		(ptr)[1] = (uint8_t)(((uint64_t)(u)) >> 48); \
		(ptr)[2] = (uint8_t)(((uint64_t)(u)) >> 40); \
		(ptr)[3] = (uint8_t)(((uint64_t)(u)) >> 32); \
		(ptr)[4] = (uint8_t)(((uint64_t)(u)) >> 24); \
		(ptr)[5] = (uint8_t)(((uint64_t)(u)) >> 16); \
		(ptr)[6] = (uint8_t)(((uint64_t)(u)) >>  8); \
		(ptr)[7] = (uint8_t)((uint64_t)(u)); \
	} while(0)

#define GET64(ptr, v) \
	v = (((uint64_t)((uint8_t)(ptr)[0])) << 56) + \
	    (((uint64_t)((uint8_t)(ptr)[1])) << 48) + \
	    (((uint64_t)((uint8_t)(ptr)[2])) << 40) + \
	    (((uint64_t)((uint8_t)(ptr)[3])) << 32) + \
	    (((uint64_t)((uint8_t)(ptr)[4])) << 24) + \
	    (((uint64_t)((uint8_t)(ptr)[5])) << 16) + \
	    (((uint64_t)((uint8_t)(ptr)[6])) <<  8) + \
	    (((uint64_t)(uint8_t)(ptr)[7]))

void showdate(time_t now) {
	printf("%s", asctime(localtime(&now)));
}

void server(const char* url) {
	auto sock = nng::rep::open();
	sock.listen(url);
	while(true) {
		uint64_t val;
		auto buf = sock.recv();
		if( buf.size() == sizeof(uint64_t) && ((GET64(buf.data<char>(), val)) == DATECMD) ) {
			printf("SERVER: RECEIVED DATE REQUEST\n");
			auto now = time(nullptr);
			printf("SERVER: SENDING DATE: ");
			showdate(now);

			// Reuse the buffer.  We know it is big enough.
			PUT64(buf.data<char>(), (uint64_t)now);
			sock.send( std::move(buf) );
		}
	}
}

void client(const char* url) {
	auto sock = nng::req::open();
	sock.dial(url);

	printf("CLIENT: SENDING DATE REQUEST\n");
	
	uint8_t cmd[sizeof(uint64_t)];
	PUT64(cmd,DATECMD);

	sock.send( cmd );
	auto buf = sock.recv();

	if(buf.size() == sizeof(uint64_t)) {
		uint64_t now;
		GET64(buf.data<char>(), now);
		printf("CLIENT: RECEIVED DATE: ");
		showdate((time_t)now);
	} else {
		printf("CLIENT: GOT WRONG SIZE!\n");
	}
}

int main( int argc, char** argv ) try {
	if(argc > 1 && strcmp(CLIENT, argv[1]) == 0) {
		client(argv[2]);
		return 0;
	}

	if(argc > 1 && strcmp(SERVER, argv[1]) == 0) {
		server(argv[2]);
		return 0;
	}

	fprintf(stderr, "Usage: reqrep %s|%s <URL> ...\n", CLIENT, SERVER);
	return 1;
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}

