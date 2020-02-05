#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>

namespace {

uint8_t dat123[] = { 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3 };

}

TEST_CASE("Lengths are empty","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE(msg.body().size() == 0);
	REQUIRE(msg.header().size() == 0);
}

TEST_CASE("We can append to the header","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.header().append( "pad" ));
	REQUIRE(msg.header().size() == 4);
	REQUIRE(msg.header().get() == "pad");
}

TEST_CASE("We can append to the body","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.body().append( "123" ));
	REQUIRE(msg.body().size() == 4);
	REQUIRE(msg.body().get() == "123");
}

TEST_CASE("We can delete from the header front","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.header().append( "def" ));
	REQUIRE_NOTHROW(msg.header().insert( {"abc",3} ));
	REQUIRE(msg.header().size() == 7);
	REQUIRE(msg.header().get() == "abcdef");
	REQUIRE_NOTHROW(msg.header().trim(2));
	REQUIRE(msg.header().size() == 5);
	REQUIRE(msg.header().get() == "cdef");
}

TEST_CASE("We can delete from the header back","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.header().append( "def" ));
	REQUIRE_NOTHROW(msg.header().insert( {"abc",3} ));
	REQUIRE(msg.header().size() == 7);
	REQUIRE(msg.header().get() == "abcdef");
	REQUIRE_NOTHROW(msg.header().chop(5));
	REQUIRE(msg.header().size() == 2);
	REQUIRE(msg.header().get() == nng::view("ab",2));
}

TEST_CASE("We can delete from the body front","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.body().append( "xyz" ));
	REQUIRE_NOTHROW(msg.body().insert( {"uvw",3} ));
	REQUIRE(msg.body().size() == 7);
	REQUIRE(msg.body().get() == "uvwxyz");
	REQUIRE_NOTHROW(msg.body().trim(2));
	REQUIRE(msg.body().size() == 5);
	REQUIRE(msg.body().get() == "wxyz");
}

TEST_CASE("We can delete from the body back","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.body().append( "xyz" ));
	REQUIRE_NOTHROW(msg.body().insert( {"uvw",3} ));
	REQUIRE(msg.body().size() == 7);
	REQUIRE(msg.body().get() == "uvwxyz");
	REQUIRE_NOTHROW(msg.body().chop(5));
	REQUIRE(msg.body().size() == 2);
	REQUIRE(msg.body().get() == nng::view("uv",2));
}

TEST_CASE("Clearing the header works","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.header().append( "bogus" ));
	REQUIRE(msg.header().size() == 6);
	REQUIRE_NOTHROW(msg.header().clear());
	REQUIRE(msg.header().size() == 0);
}

TEST_CASE("Clearing the body works","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.body().append( "bogus" ));
	REQUIRE(msg.body().size() == 6);
	REQUIRE_NOTHROW(msg.body().clear());
	REQUIRE(msg.body().size() == 0);
}

TEST_CASE("We cannot delete more header than exists","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.header().append( "short" ));
	REQUIRE(msg.header().size() == 6);
	REQUIRE_THROWS_WITH(
		msg.header().trim(16),
		nng::exception(nng::error::inval).what()
	);
	REQUIRE(msg.header().size() == 6);
	REQUIRE_THROWS_WITH(
		msg.header().chop(16),
		nng::exception(nng::error::inval).what()
	);
	REQUIRE(msg.header().size() == 6);
	REQUIRE(msg.header().get() == "short");
}

TEST_CASE("We cannot delete more body than exists","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.body().append( "short" ));
	REQUIRE(msg.body().size() == 6);
	REQUIRE_THROWS_WITH(
		msg.body().trim(16),
		nng::exception(nng::error::inval).what()
	);
	REQUIRE(msg.body().size() == 6);
	REQUIRE_THROWS_WITH(
		msg.body().chop(16),
		nng::exception(nng::error::inval).what()
	);
	REQUIRE(msg.body().size() == 6);
	REQUIRE(msg.body().get() == "short");
}

TEST_CASE("Pipe retrievals work","[message]") {
	auto msg = nng::make_msg(0);
	nng::pipe_view p0;

	REQUIRE(p0.id() < 0);
	nng::pipe_view p = msg.get_pipe();
	REQUIRE(p.id() < 0);
	p = nng::pipe_view({0x22222222});
	msg.set_pipe(p);
	p = msg.get_pipe();
	REQUIRE(p.id() != p0.id());
	REQUIRE(p.id() == 0x22222222);
}

TEST_CASE("Message realloc works","[message]") {
	auto msg = nng::make_msg(0);
	REQUIRE_NOTHROW(msg.body().append( "abc" ));
	REQUIRE_NOTHROW(msg.realloc( 1500 ));
	REQUIRE(msg.body().size() == 1500);
	REQUIRE(nng::view(msg.body().get().data(),4) == "abc");
	REQUIRE_NOTHROW(msg.realloc( 2 ));
	REQUIRE(msg.body().size() == 2);
	REQUIRE(msg.body().get() == nng::view("ab",2));
	REQUIRE_NOTHROW(msg.body().append( "CDEF" ));
	REQUIRE(msg.body().size() == 7);
	REQUIRE(msg.body().get() == "abCDEF");
}

TEST_CASE("Inserting a lot of data works","[message]") {
	auto msg = nng::make_msg(0);
	char chunk[1024];
	memset(chunk, '+', sizeof(chunk));
			
	REQUIRE_NOTHROW(msg.body().append( "abc" ));
	REQUIRE(msg.body().size() == 4);
	REQUIRE_NOTHROW(msg.body().insert( chunk ));
	REQUIRE(msg.body().size() == sizeof(chunk) + 4);
			
	REQUIRE(nng::view(msg.body().data(),sizeof(chunk)) == chunk);
	REQUIRE(nng::view(msg.body().data<char>() + sizeof(chunk),4) == "abc");
			
	REQUIRE_NOTHROW(msg.body().trim( sizeof(chunk) - 2 ));
			
	REQUIRE(msg.body().get() == "++abc");
}

TEST_CASE("Message dup works","[message]") {
	auto msg = nng::make_msg(0);
			
	REQUIRE_NOTHROW(msg.header().append( "front" ));
	REQUIRE_NOTHROW(msg.body().append( "back" ));

	auto m2 = msg;
			
	REQUIRE(msg.body().size() == 5);
	REQUIRE(m2.body().size() == 5);
	REQUIRE(msg.header().size() == m2.header().size());
			
	REQUIRE_NOTHROW(msg.body().insert( {"way",3} ));
			
	REQUIRE(msg.body().size() == 8);
	REQUIRE(m2.body().size() == 5);
			
	REQUIRE(msg.body().get() == "wayback");
	REQUIRE(m2.body().get() == "back");
			
	REQUIRE_NOTHROW(m2.body().chop( 1 ));
	REQUIRE_NOTHROW(m2.body().append( "2basics" ));
	REQUIRE(msg.body().size() == 8);
	REQUIRE(msg.body().get() == "wayback");
	REQUIRE(m2.body().size() == 12);
	REQUIRE(m2.body().get() == "back2basics");
}

TEST_CASE("Missing option fails properly","[message]") {
	auto msg = nng::make_msg(0);
	nng::buffer out(128);
	REQUIRE_THROWS_WITH(
		msg.get_opt(4545, out),
		nng::exception(nng::error::noent).what()
	);
}

TEST_CASE("Uint32 body operations work","[message]") {
	auto msg = nng::make_msg(0);

	REQUIRE_NOTHROW(msg.body().append_u32(2));
	REQUIRE_NOTHROW(msg.body().insert_u32(1));
	REQUIRE_NOTHROW(msg.body().append_u32(3));
	REQUIRE_NOTHROW(msg.body().insert_u32(0));
	REQUIRE(msg.body().trim_u32() == 0);
				
	REQUIRE(msg.body().size() == sizeof(dat123));
	REQUIRE(msg.body().get() == dat123);
				
	REQUIRE(msg.body().trim_u32() == 1);
	REQUIRE(msg.body().chop_u32() == 3);
	REQUIRE(msg.body().trim_u32() == 2);
				
	REQUIRE_THROWS_WITH(
		msg.body().trim_u32(),
		nng::exception(nng::error::inval).what()
	);
	REQUIRE_THROWS_WITH(
		msg.body().trim_u32(),
		nng::exception(nng::error::inval).what()
	);
				
	SECTION("Single byte is inadequate") {
		msg.body().clear();
		uint32_t v = 2;

		REQUIRE_NOTHROW(msg.body().append( {&v,1} ));

		REQUIRE_THROWS_WITH(
			msg.body().trim_u32(),
			nng::exception(nng::error::inval).what()
		);
		REQUIRE_THROWS_WITH(
			msg.body().trim_u32(),
			nng::exception(nng::error::inval).what()
		);
	}
}

TEST_CASE("Uint32 header operations work","[message]") {
	auto msg = nng::make_msg(0);

	REQUIRE_NOTHROW(msg.header().append_u32(2));
	REQUIRE_NOTHROW(msg.header().insert_u32(1));
	REQUIRE_NOTHROW(msg.header().append_u32(3));
	REQUIRE_NOTHROW(msg.header().insert_u32(0));
	REQUIRE(msg.header().trim_u32() == 0);
				
	REQUIRE(msg.header().size() == sizeof(dat123));
	REQUIRE(msg.header().get() == dat123);
				
	REQUIRE(msg.header().trim_u32() == 1);
	REQUIRE(msg.header().chop_u32() == 3);
	REQUIRE(msg.header().trim_u32() == 2);
				
	REQUIRE_THROWS_WITH(
		msg.header().trim_u32(),
		nng::exception(nng::error::inval).what()
	);
	REQUIRE_THROWS_WITH(
		msg.header().trim_u32(),
		nng::exception(nng::error::inval).what()
	);
	
	SECTION("Single byte is inadequate") {
		msg.header().clear();
		uint32_t v = 2;

		REQUIRE_NOTHROW(msg.header().append( {&v,1} ));

		REQUIRE_THROWS_WITH(
			msg.header().trim_u32(),
			nng::exception(nng::error::inval).what()
		);
		REQUIRE_THROWS_WITH(
			msg.header().trim_u32(),
			nng::exception(nng::error::inval).what()
		);
	}
}
