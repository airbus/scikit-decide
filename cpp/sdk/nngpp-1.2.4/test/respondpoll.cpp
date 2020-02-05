#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/survey0.h>
#include <nngpp/protocol/respond0.h>
#include <nngpp/platform/platform.h>

TEST_CASE("Respondent pollable") {
	auto addr = "inproc://respondpoll";

	SECTION("Given a connected survey pair") {
		auto surv = nng::survey::open();
		REQUIRE(surv);
		auto resp = nng::respond::open();
		REQUIRE(resp);
		nng::ctx ctx( resp );
		REQUIRE(ctx);

		REQUIRE_NOTHROW( nng::set_opt_send_timeout(surv,2000) );
		REQUIRE_NOTHROW( nng::set_opt_send_timeout(resp,2000) );
		REQUIRE_NOTHROW( nng::set_opt_recv_timeout(surv,2000) );
		REQUIRE_NOTHROW( nng::set_opt_recv_timeout(resp,2000) );

		REQUIRE_NOTHROW( resp.listen(addr) );

		SECTION("Respondent ctx not pollable") {
			REQUIRE_THROWS_WITH(
				ctx.get_opt_int( NNG_OPT_SENDFD ),
				nng::exception(nng::error::notsup).what()
			);
			REQUIRE_THROWS_WITH(
				ctx.get_opt_int( NNG_OPT_RECVFD ),
				nng::exception(nng::error::notsup).what()
			);
		}

		SECTION("Respondent starts not writable") {
			nng::get_opt_send_fd(resp);

			SECTION("And remains unwritable on connect") {
				REQUIRE_NOTHROW( surv.dial(addr) );

				nng::msleep(100);

				SECTION("Becomes writable after recv") {
					surv.send( nng::make_msg(0) );
					resp.recv_msg();
				}
			}
		}

		SECTION("Respondent starts not readable") {
			nng::get_opt_recv_fd(resp);

			SECTION("And doesn't become readable on connect") {
				REQUIRE_NOTHROW( surv.dial(addr) );
				nng::msleep(100);
			}

			SECTION("And becomes readable on data") {
				REQUIRE_NOTHROW( surv.dial(addr) );
				nng::msleep(200);

				auto msg = nng::make_msg(0);
				msg.body().append( {"xyz",3} );
				surv.send( std::move(msg) );

				nng::msleep(300); // give time for msg to arrive
				
				SECTION("Is no longer readable after recv") {
					msg = resp.recv_msg();
				}
			}
		}
	}
}
