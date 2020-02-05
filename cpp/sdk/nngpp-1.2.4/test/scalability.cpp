#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/req0.h>
#include <nngpp/protocol/rep0.h>
#include <nngpp/platform/platform.h>
#include <vector>

namespace {

static const char* addr = "inproc://scalability";

struct server {
	nng::socket rep = nng::rep::open();

	void run() const {
		nng::set_opt_recv_buffer(rep,256);
		nng::set_opt_send_buffer(rep,256);
		rep.listen(addr);

		while(true) {
			try {
				rep.send( rep.recv_msg() );
			}
			catch( const nng::exception& ) {
				break;
			}
		}
	}

	void stop() {
		rep = nng::socket();
	}

};

static void serve(void* arg) {
	((server*)arg)->run();
}

static void open_clients(nng::socket* clients, int num) {
	nng_duration t = 100; // 100ms
	for(int i=0;i<num;++i) {
		CAPTURE(i);
		auto c = nng::req::open();
		REQUIRE_NOTHROW(nng::set_opt_recv_timeout(c,t));
		REQUIRE_NOTHROW(nng::set_opt_send_timeout(c,t));
		nng::msleep(10);
		REQUIRE_NOTHROW(c.dial(addr));
		clients[i] = std::move(c);
	}
}

static void transact(nng::socket* clients, int num) {
	for(int i=0;i<num;++i) {
		CAPTURE(i);
		clients[i].send( nng::make_msg(0) );
		clients[i].recv_msg();
	}
}

}

TEST_CASE("Scalability","[scalability]") {
	int nclients = 200;
	server sv;
	nng::thread server(serve, &sv);
	std::vector<nng::socket> clients(nclients);

	REQUIRE(server);

	REQUIRE_NOTHROW(open_clients(clients.data(), nclients));

	REQUIRE_NOTHROW(transact(clients.data(), nclients));

	sv.stop();
}

