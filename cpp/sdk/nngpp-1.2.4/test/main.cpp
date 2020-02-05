#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_COLOUR_NONE
#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>

int main( int argc, char* argv[] ) {
	int result = Catch::Session().run( argc, argv );
	nng::fini();
	return result;
}
