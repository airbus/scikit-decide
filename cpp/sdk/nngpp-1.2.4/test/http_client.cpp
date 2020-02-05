#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/http/http.h>
#include <nngpp/platform/platform.h>

TEST_CASE("HTTP Client","[httpclient]") {

	SECTION("Given a TCP connection to httpbin.org") {

		auto aio = nng::make_aio();
		REQUIRE(aio);

		nng::url url("http://example.org/");
		REQUIRE(url);

		nng::http::client cli(url);
		REQUIRE(cli);

		cli.connect(aio);
		aio.wait();
		
		REQUIRE( aio.result() == nng::error::success );

		nng::http::conn_view http = aio.get_output<nng_http_conn>(0);

		SECTION("We can initiate a message") {
			REQUIRE(http);

			nng::http::req req(url);
			REQUIRE(req);
			auto res = nng::http::make_res();
			REQUIRE(res);

			http.write(req,aio);
			aio.wait();

			REQUIRE( aio.result() == nng::error::success );
			
			http.read(res,aio);
			aio.wait();
			
			REQUIRE( aio.result() == nng::error::success );
			REQUIRE( res.get_status() == nng::http::status::ok );

			SECTION("The message contents are correct") {
				auto cstr = res.get_header("Content-Length");
				REQUIRE(cstr);
				auto sz = atoi(cstr);
				REQUIRE(sz > 0);

				nng::buffer data(sz);
				REQUIRE(data);

				nng_iov iov = data;
				REQUIRE_NOTHROW( aio.set_iov(iov) );

				aio.wait();
				REQUIRE( aio.result() == nng::error::success );

				http.read_all(aio);
				aio.wait();
				REQUIRE( aio.result() == nng::error::success );

			}
		}
	}
}
