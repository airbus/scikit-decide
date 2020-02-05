#include <catch2/catch.hpp>
#include <cstring>
#include <nngpp/nngpp.h>
#include <nngpp/transport/tcp.h>

inline uint16_t test_htons( uint16_t in ) noexcept {
	in = ((in >> 8) & 0xff) | ((in & 0xff) << 8);
	return in;
}

TEST_CASE("Supplemental TCP", "[tcpapi]") {
	
	INFO("Listener listens (wildcard)");
	auto l = nng::stream::listener("tcp://127.0.0.1");
	REQUIRE(l);
	REQUIRE_NOTHROW(l.listen());

	uint8_t ip[4];
	ip[0] = 127;
	ip[1] = 0;
	ip[2] = 0;
	ip[3] = 1;

	auto sa = l.get_addr( to_name(nng::option::local_address) );

	REQUIRE(sa.s_in.sa_port != 0);
	REQUIRE(memcmp(&sa.s_in.sa_addr, ip, 4) == 0);

	INFO("We can dial it");
	nng::aio daio(nullptr,nullptr);
	nng::aio laio(nullptr,nullptr);
	nng::aio maio(nullptr,nullptr);

	char uri[64];
	snprintf(uri, sizeof(uri), "tcp://127.0.0.1:%d", test_htons(sa.s_in.sa_port));
	auto d = nng::stream::dialer(uri);
	REQUIRE(d);
	
	d.dial(daio);
	l.accept(laio);

	daio.wait();
	REQUIRE(daio.result() == nng::error::success);
	laio.wait();
	REQUIRE(laio.result() == nng::error::success);

	nng::stream::stream c1( daio.get_output<nng_stream>(0) );
	nng::stream::stream c2( laio.get_output<nng_stream>(0) );
	REQUIRE(c1);
	REQUIRE(c2);
	
	INFO("They exchange messages");
	nng::aio aio1(nullptr,nullptr);
	nng::aio aio2(nullptr,nullptr);

	REQUIRE_NOTHROW(c1.set_bool( to_name(nng::tcp::option::no_delay), true) );
	REQUIRE_NOTHROW(c2.set_bool( to_name(nng::tcp::option::no_delay), true) );
	REQUIRE_NOTHROW(c1.set_bool( to_name(nng::tcp::option::keep_alive), true) );
	
	char buf1[5];
	memcpy(buf1, "TEST", 5);

	char buf2[5];
	memset(buf2, 0, 5);
	
	aio1.set_iov( {buf1, 5} );
	aio2.set_iov( {buf2, 5} );
	c1.send(aio1);
	c2.recv(aio2);
	aio1.wait();
	aio2.wait();

	REQUIRE(aio1.result() == nng::error::success);
	REQUIRE(aio1.count() == 5);

	REQUIRE(aio2.result() == nng::error::success);
	REQUIRE(aio2.count() == 5);

	REQUIRE(memcmp(buf1, buf2, 5) == 0);

	INFO("Socket name matches");
	{
		nng_sockaddr sa2;
		REQUIRE_NOTHROW(sa2 = c2.get_addr( to_name(nng::option::local_address) ));
		REQUIRE(sa2.s_in.sa_family == NNG_AF_INET);
		REQUIRE(sa2.s_in.sa_addr == sa.s_in.sa_addr);
		REQUIRE(sa2.s_in.sa_port == sa.s_in.sa_port);
	}
	
	INFO("Peer name matches")
	{
		nng_sockaddr sa2;
		REQUIRE_NOTHROW(sa2 = c1.get_addr( to_name(nng::option::remote_address) ));
		REQUIRE(sa2.s_in.sa_family == NNG_AF_INET);
		REQUIRE(sa2.s_in.sa_addr == sa.s_in.sa_addr);
		REQUIRE(sa2.s_in.sa_port == sa.s_in.sa_port);
	}

}
