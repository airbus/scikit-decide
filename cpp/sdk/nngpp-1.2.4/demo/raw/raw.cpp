// This is a port of the nng demo to nngpp
// See https://github.com/nanomsg/nng/tree/master/demo/raw

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/req0.h>
#include <nngpp/protocol/rep0.h>
#include <nngpp/platform/platform.h>

// Parallel is the maximum number of outstanding requests we can handle.
// This is *NOT* the number of threads in use, but instead represents
// outstanding work items.  Select a small number to reduce memory size.
// (Each one of these can be thought of as a request-reply loop.)
#ifndef PARALLEL
#define PARALLEL 32
#endif

void server_cb(void* arg);

// The server keeps a list of work items, sorted by expiration time,
// so that we can use this to set the timeout to the correct value for
// use in poll.
struct work {
	enum { INIT, RECV, WAIT, SEND } state = INIT;
	nng::aio aio{ server_cb, this };
	nng::socket_view sock;
	nng::msg msg;

	explicit work( nng::socket_view sock ) : sock(sock) {}
};

void server_cb(void* arg) {
	work* work = (struct work*)arg;
	uint32_t when;

	switch(work->state) {
	case work::INIT:
		work->state = work::RECV;
		work->sock.recv( work->aio );
		break;
	case work::RECV:
		{
			auto result = work->aio.result();
			if( result != nng::error::success ) {
				throw nng::exception(result);
			}
		}
		{
			auto msg = work->aio.release_msg();
			try {
				when = msg.body().trim_u32();
			}
			catch( const nng::exception& ) {
				// bad message, just ignore it.
				work->sock.recv( work->aio );
				return;
			}

			work->msg = std::move(msg);
		}
		work->state = work::WAIT;
		nng::sleep( when, work->aio );
		break;
	case work::WAIT:
		// We could add more data to the message here.
		work->aio.set_msg( std::move(work->msg) );
		work->state = work::SEND;
		work->sock.send( work->aio );
		break;
	case work::SEND:
		{
			auto result = work->aio.result();
			if( result != nng::error::success ) {
				throw nng::exception(result);
			}
		}
		work->state = work::RECV;
		work->sock.recv( work->aio );
		break;
	default:
		throw nng::exception(nng::error::state);
		break;
	}
}

// The server runs forever.
void server(const char* url) {

	//  Create the socket.
	auto sock = nng::rep::open_raw();

	std::unique_ptr<work> works[PARALLEL];
	for (int i=0;i<PARALLEL;++i) {
		works[i] = std::make_unique<work>(sock);
	}

	sock.listen(url);

	for(int i=0;i<PARALLEL;++i) {
		server_cb(works[i].get()); // this starts them going (INIT state)
	}

	while(true) {
		nng::msleep(3'600'000); // neither pause() nor sleep() portable
	}
}

//  The client runs just once, and then returns.
void client(const char* url, const char* msecstr) {
	auto msec = atoi(msecstr) * 1000;

	auto sock = nng::req::open();
	sock.dial(url);
	
	auto start = nng::clock();
	
	auto msg = nng::make_msg(0);
	msg.body().append_u32(msec);
	sock.send( std::move(msg) );
	msg = sock.recv_msg();

	auto end = nng::clock();

	printf("Request took %u milliseconds.\n", (uint32_t)(end - start));
}

int main(int argc, char** argv) try {
	if(argc < 3) {
		fprintf(stderr, "Usage: %s <url> [-s|<secs>]\n", argv[0]);
		return 1;
	}
	if(strcmp(argv[2], "-s") == 0) {
		server(argv[1]);
	}
	else {
		client(argv[1], argv[2]);
	}
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}

