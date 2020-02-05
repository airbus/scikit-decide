#include <catch2/catch.hpp>
#include <cstring>
#include <nngpp/nngpp.h>

TEST_CASE("Supplemental IPC", "[ipc]") {
	
	INFO("We can create a dialer and listener");
	auto url = "ipc:///tmp/ipcsupp_test";

	auto d = nng::stream::dialer(url);
	REQUIRE(d);
	auto l = nng::stream::listener(url);
	REQUIRE(l);
	
	INFO("Listener listens");
	REQUIRE_NOTHROW(l.listen());

	INFO("We can dial it");
	nng::aio daio(nullptr,nullptr);
	nng::aio laio(nullptr,nullptr);
	nng::aio maio(nullptr,nullptr);

	d.dial(daio);
	l.accept(laio);

	daio.wait();
	laio.wait();

	REQUIRE(daio.result() == nng::error::success);
	REQUIRE(laio.result() == nng::error::success);

	nng::stream::stream c1( daio.get_output<nng_stream>(0) );
	nng::stream::stream c2( laio.get_output<nng_stream>(0) );
	REQUIRE(c1);
	REQUIRE(c2);
	
	INFO("They exchange messages");
	nng::aio aio1(nullptr,nullptr);
	nng::aio aio2(nullptr,nullptr);
	
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
		REQUIRE_NOTHROW(sa2 = c2.get_addr(to_name(nng::option::local_address)));
		REQUIRE(sa2.s_ipc.sa_family == NNG_AF_IPC);
		REQUIRE(strcmp(sa2.s_ipc.sa_path, url + strlen("ipc://")) == 0);
	}

}
