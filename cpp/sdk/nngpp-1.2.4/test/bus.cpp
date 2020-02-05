#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/bus0.h>

TEST_CASE("BUS pattern","[bus]") {
	auto addr = "inproc://bus";

	SECTION("We can create a BUS socket") {
		auto bus = nng::bus::open();
		REQUIRE(bus);
	}

	SECTION("We can create a linked BUS topology") {

		auto bus1 = nng::bus::open();
		REQUIRE(bus1);
		auto bus2 = nng::bus::open();
		REQUIRE(bus2);
		auto bus3 = nng::bus::open();
		REQUIRE(bus3);

		REQUIRE_NOTHROW( bus1.listen(addr) );
		REQUIRE_NOTHROW( bus2.dial(addr) );
		REQUIRE_NOTHROW( bus3.dial(addr) );

		nng_duration rtimeo = 50;
		REQUIRE_NOTHROW( nng::set_opt_recv_timeout( bus1, rtimeo ) );
		REQUIRE_NOTHROW( nng::set_opt_recv_timeout( bus2, rtimeo ) );
		REQUIRE_NOTHROW( nng::set_opt_recv_timeout( bus3, rtimeo ) );

		SECTION("Messages delivered") {
			auto expected_error = nng::exception(nng::error::timedout).what();

			REQUIRE_THROWS_WITH( bus1.recv_msg(), expected_error );
			REQUIRE_THROWS_WITH( bus2.recv_msg(), expected_error );
			REQUIRE_THROWS_WITH( bus3.recv_msg(), expected_error );
			
			{
				auto msg = nng::make_msg(0);
				REQUIRE(msg);

				msg.body().append("99bits");
				REQUIRE_NOTHROW( bus2.send( std::move(msg) ) );
				REQUIRE_NOTHROW( msg = bus1.recv_msg() );
				REQUIRE( msg.body().get() == "99bits" );

				REQUIRE_THROWS_WITH( msg = bus3.recv_msg(), expected_error );
			}
			{
				auto msg = nng::make_msg(0);
				REQUIRE(msg);

				msg.body().append("onthe");
				REQUIRE_NOTHROW( bus1.send( std::move(msg) ) );

				REQUIRE_NOTHROW( msg = bus2.recv_msg() );
				REQUIRE( msg.body().get() == "onthe" );
				REQUIRE_NOTHROW( msg = bus3.recv_msg() );
				REQUIRE( msg.body().get() == "onthe" );
			}
		}
	}
}
