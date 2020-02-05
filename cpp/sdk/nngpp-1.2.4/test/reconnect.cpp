#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/push0.h>
#include <nngpp/protocol/pull0.h>
#include <nngpp/platform/platform.h>
#include <cstring>

TEST_CASE("Reconnect works","[reconnect]") {
	auto addr = "inproc://reconnect";

	SECTION("Dialing before listening works") {
		auto push = nng::push::open();
		REQUIRE(push);
		auto pull = nng::pull::open();
		REQUIRE(pull);

		REQUIRE_NOTHROW( nng::set_opt_reconnect_time_min( pull, 10 ) );
		REQUIRE_NOTHROW( nng::set_opt_reconnect_time_max( pull, 10 ) );

		REQUIRE_NOTHROW( push.dial(addr,nng::flag::nonblock) );
		nng::msleep(100);
		REQUIRE_NOTHROW( pull.listen(addr) );

		SECTION("We can send a frame") {
			nng::msleep(100);

			auto msg = nng::make_msg(0);
			REQUIRE(msg);
			msg.body().append("hello");
			REQUIRE_NOTHROW( push.send( std::move(msg) ) );
			REQUIRE_NOTHROW( msg = pull.recv_msg() );

			REQUIRE(msg);
			REQUIRE( msg.body().get() == "hello" );
		}
	}

	SECTION("Reconnection works") {
		auto push = nng::push::open();
		REQUIRE(push);
		auto pull = nng::pull::open();
		REQUIRE(pull);

		REQUIRE_NOTHROW( nng::set_opt_reconnect_time_min( pull, 10 ) );
		REQUIRE_NOTHROW( nng::set_opt_reconnect_time_max( pull, 10 ) );

		REQUIRE_NOTHROW( push.dial(addr,nng::flag::nonblock) );
		{
			nng::listener l( pull, addr, 0 );
			REQUIRE(l);
			nng::msleep(100);
		}
		pull.listen(addr);

		SECTION("They still exchange frames") {
			nng::pipe_view p1;
			nng::msleep(100);
			{
				auto msg = nng::make_msg(0);
				REQUIRE(msg);
				msg.body().append("hello");
				REQUIRE_NOTHROW( push.send( std::move(msg) ) );
				REQUIRE_NOTHROW( msg = pull.recv_msg() );
				REQUIRE(msg);
				REQUIRE( msg.body().get() == "hello" );
				
				p1 = msg.get_pipe();
			}
			SECTION("Even after pipe close") {
				nng::pipe_view p2;
				nng_pipe_close(p1.get());
				nng::msleep(100);
				{
					auto msg = nng::make_msg(0);
					REQUIRE(msg);
					msg.body().append("again");
					REQUIRE_NOTHROW( push.send( std::move(msg) ) );
					REQUIRE_NOTHROW( msg = pull.recv_msg() );
					REQUIRE(msg);
					REQUIRE( msg.body().get() == "again" );

					p2 = msg.get_pipe();
				}
				REQUIRE(p2.id() != p1.id());
			}
		}
	}
}
