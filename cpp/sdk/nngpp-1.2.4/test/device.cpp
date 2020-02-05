#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/pair1.h>
#include <nngpp/platform/platform.h>
#include <thread>

TEST_CASE("We cannot create cooked mode device","[device]") {
	auto s1 = nng::pair::v1::open();
	REQUIRE(s1);
	REQUIRE_THROWS_WITH(
		nng::device(s1,s1),
		nng::exception(nng::error::inval).what()
	);
}

TEST_CASE("PAIRv1 device","[device]") {
	auto addr1 = "inproc://dev1";
	auto addr2 = "inproc://dev2";

	SECTION("We can create a PAIRv1 device") {
		std::thread thr;
		{
			auto dev1 = nng::pair::v1::open_raw();
			REQUIRE(dev1);
			auto dev2 = nng::pair::v1::open_raw();
			REQUIRE(dev2);

			thr = std::thread([&] {
				try { nng::device(dev1,dev2); }
				catch(...) {}
			});

			REQUIRE_NOTHROW( dev1.listen(addr1) );
			REQUIRE_NOTHROW( dev2.listen(addr2) );

			auto end1 = nng::pair::v1::open();
			REQUIRE(end1);
			auto end2 = nng::pair::v1::open();
			REQUIRE(end2);

			REQUIRE_NOTHROW( end1.dial(addr1) );
			REQUIRE_NOTHROW( end2.dial(addr2) );

			nng_duration tmo = 1000;
			REQUIRE_NOTHROW( nng::set_opt_recv_timeout(end1,tmo) );
			REQUIRE_NOTHROW( nng::set_opt_recv_timeout(end2,tmo) );

			nng::msleep(100);
			SECTION("Device can send and receive") {
				{
					auto msg = nng::make_msg(0);
					REQUIRE(msg);

					msg.body().append("ALPHA");
					REQUIRE_NOTHROW( end1.send(std::move(msg)) );
					REQUIRE_NOTHROW( msg = end2.recv_msg() );
					REQUIRE( msg.body().get() == "ALPHA" );
				}
				{
					auto msg = nng::make_msg(0);
					REQUIRE(msg);

					msg.body().append("OMEGA");
					REQUIRE_NOTHROW( end2.send(std::move(msg)) );
					REQUIRE_NOTHROW( msg = end1.recv_msg() );
					REQUIRE( msg.body().get() == "OMEGA" );
				}
			}
		}
		thr.join();
	}
}
