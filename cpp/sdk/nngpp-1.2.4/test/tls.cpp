#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/pair1.h>
#include <nngpp/transport/tls.h>
#include <nngpp/transport/tcp.h>
#include <nngpp/platform/platform.h>

namespace {

static const char cert[] =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIDLjCCAhYCCQDtskvsw6K+mDANBgkqhkiG9w0BAQsFADBYMQswCQYDVQQGEwJV\n"
    "UzELMAkGA1UECAwCQ0ExEjAQBgNVBAcMCVNhbiBEaWVnbzEUMBIGA1UECgwLbmFu\n"
    "b21zZy5vcmcxEjAQBgNVBAMMCWxvY2FsaG9zdDAgFw0xODAxMTEyMjM0MzVaGA8y\n"
    "MTE3MTIxODIyMzQzNVowWDELMAkGA1UEBhMCVVMxCzAJBgNVBAgMAkNBMRIwEAYD\n"
    "VQQHDAlTYW4gRGllZ28xFDASBgNVBAoMC25hbm9tc2cub3JnMRIwEAYDVQQDDAls\n"
    "b2NhbGhvc3QwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDMvoHdEnfO\n"
    "hmG3PTj6YC5qz6N5hgmcwf4EZkor4+R1Q5hDOKqOknWmVuGBD5mA61ObK76vycIT\n"
    "Tp+H+vKvfgunySZrlyYg8IbgoDbvVgj9RF8xFHdN0PVeqnkBCsCzLtSu6TP8PSgI\n"
    "SKiRMH0NUSakWqCPEc2E1r1CKdOpa7av/Na30LPsuKFcAUhu7QiVYfER86ktrO8G\n"
    "F2PeVy44Q8RkiLw8uhU0bpAflqkR1KCjOLajw1eL3C+Io75Io8qUOLxWc3LH0hl3\n"
    "oEI0jWu7JYlRAw/O7xm4pcGTwy5L8Odz4a7ZTAmuapFRarGOIcDg8Yr0tllRd1mH\n"
    "1T4Z2Wv7Rs0tAgMBAAEwDQYJKoZIhvcNAQELBQADggEBAIfUXK7UonrYAOrlXUHH\n"
    "gfHNdOXMzQP2Ms6Sxov+1tCTfgsYE65Mggo7hRJUqmKpstpbdRBVXhTyht/xjyTz\n"
    "5sMjoeCyv1tXOHpLTfD3LBXwYZwsFdoLS1UHhD3qiYjCyyY2LWa6S786CtlcbCvu\n"
    "Uij2q8zJ4WFrNqAzxZtsTfg16/6JRFw9zpVSCNlHqCxNQxzWucbmUFTiWn9rnc/N\n"
    "r7utG4JsDPZbEI6QS43R7gGLDF7s0ftWKqzlQiZEtuDQh2p7Uejbft8XmZd/VuV/\n"
    "dFMXOO1rleU0lWAJcXWOWHH3er0fivu2ISL8fRjjikYvhRGxtkwC0kPDa2Ntzgd3\n"
    "Hsg=\n"
    "-----END CERTIFICATE-----\n";

static const char key[] =
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEpQIBAAKCAQEAzL6B3RJ3zoZhtz04+mAuas+jeYYJnMH+BGZKK+PkdUOYQziq\n"
    "jpJ1plbhgQ+ZgOtTmyu+r8nCE06fh/ryr34Lp8kma5cmIPCG4KA271YI/URfMRR3\n"
    "TdD1Xqp5AQrAsy7Urukz/D0oCEiokTB9DVEmpFqgjxHNhNa9QinTqWu2r/zWt9Cz\n"
    "7LihXAFIbu0IlWHxEfOpLazvBhdj3lcuOEPEZIi8PLoVNG6QH5apEdSgozi2o8NX\n"
    "i9wviKO+SKPKlDi8VnNyx9IZd6BCNI1ruyWJUQMPzu8ZuKXBk8MuS/Dnc+Gu2UwJ\n"
    "rmqRUWqxjiHA4PGK9LZZUXdZh9U+Gdlr+0bNLQIDAQABAoIBAC82HqvjfkzZH98o\n"
    "9uKFGy72AjQbfEvxT6mkDKZiPmPr2khl4K5Ph2F71zPzbOoVWYoGZEoUs/PPxWmN\n"
    "rDhbUES4VWupxtkBnZheWUyHAjukcG7Y0UnYTTwvAwgCerzWp6RNkfcwAvMmDfis\n"
    "vak8dTSg0TUsXb+r5KhFDNGcTNv3f7R0cJmaZ/t9FT7SerXf1LW7itvTjRor8/ZK\n"
    "KPwT4oklp1o6RFXSenn/e2e3rAjI+TEwJA3Zp5dqO/M/AhaZKVaxL4voDVdVVkT+\n"
    "LHJWVhjLY5ilPkmPWqmZ2reTaF+gGSSjAQ+t/ahGWFqEdWIz9UoXhBBOd1ibeyvd\n"
    "Kyxp1QECgYEA8KcDkmwPrhqFlQe/U+Md27OhrQ4cecLCa6EVLsCXN1bFyCi3NSo2\n"
    "o5zFCC699KOL0ZwSmYlaQP4xjnqv4Gsa0s3uL7tqOJR2UuEtGK/MPMluGHVaWsGt\n"
    "zbnWH3xgsvvsxdt6hInFhcABLDupW336tJ8EcH7mOKoIP+azwF4kPiUCgYEA2c09\n"
    "zJBUW6SZXhgJ5vgENYc+UwDT7pfhIWZaRL+wXnwSoa7igodTKJtQp/KfFBJK4RA0\n"
    "prvwj4Wr/1ScaboR2hYZApbqXU5zkEkjC1hHIbg1fBe0EcnhP7ojMXrk6B5ed+Lq\n"
    "OVdYhUuvtdL/perelmbTJLnb8S214+tzVyg7EGkCgYEA6JLwX8zxpnhZSztOjBr9\n"
    "2zuSb7YojQBNd0kZOLLGMaQ5xwSactYWMi8rOIo76Lc6RFxKmXnl8NP5PtKRMRkx\n"
    "tjNxE05UDNRmOhkGxUn433JoZVjc9sMhXqZQKuPAbJoOLPW9RWQEsgtq1r3eId7x\n"
    "sSfRWYs6od6p1F/4rlwNOMUCgYEAtJmqf+DCAoe3IL3gICRSISy28k7CbZqE9JQR\n"
    "j+Y/Uemh7W29pyydOROoysq1PAh7DKrKbeNzcx8NYxh+5nCC8wrVzD7lsV8nFmJ+\n"
    "655UxVIhD3f8Oa/j1lr7acEU5KCiBtkjDU8vOMBsv+FpWOQrlB1JQa/X/+G+bHLF\n"
    "XmUerNkCgYEAv7R8vIKgJ1f69imgHdB31kue3wnOO/6NlfY3GTcaZcTdChY8SZ5B\n"
    "xits8xog0VcaxXhWlfO0hyCnZ9YRQbyDu0qp5eBU2p3qcE01x4ljJBZUOTweG06N\n"
    "cL9dYcwse5FhNMjrQ/OKv6B38SIXpoKQUtjgkaMtmpK8cXX1eqEMNkM=\n"
    "-----END RSA PRIVATE KEY-----\n";


static void init_dialer_tls( nng::dialer_view d ) {
	nng::tls::config cfg( nng::tls::mode::client );
	cfg.config_ca_chain(cert)
		.config_server_name("127.0.0.1")
		.config_auth_mode(nng::tls::auth_mode::none);
	set_opt_config( d, cfg );
}

static void init_listener_tls( nng::listener_view l ) {
	nng::tls::config cfg( nng::tls::mode::server );
	cfg.config_own_cert(cert,key);
	set_opt_config( l, cfg );
}

}

TEST_CASE("We can bind to port zero", "[tls]" ) {
	REQUIRE_NOTHROW(nng::tls::register_transport());

	auto s1 = nng::pair::open();
	REQUIRE(s1);
	auto s2 = nng::pair::open();
	REQUIRE(s2);

	nng::listener l( s1, "tls+tcp://127.0.0.1:0" );
	REQUIRE(l);
	
	REQUIRE_NOTHROW(init_listener_tls(l));
	REQUIRE_NOTHROW(l.start());

	auto addr = get_opt_url(l);
	CAPTURE( addr.data<char>() );

	nng::dialer d( s2, addr.data<char>() );
	REQUIRE_NOTHROW(init_dialer_tls(d));
	REQUIRE_NOTHROW(nng::tls::set_opt_auth_mode(d,nng::tls::auth_mode::none));
	REQUIRE_NOTHROW(d.start());
}

TEST_CASE("Malformed TLS addresses do not panic", "[tls]" ) {
	REQUIRE_NOTHROW(nng::tls::register_transport());

	auto s1 = nng::pair::open();
	REQUIRE(s1);

	auto invalid_address = nng::exception(nng::error::addrinval).what();

	REQUIRE_THROWS_WITH( s1.dial("tls+tcp://127.0.0.1"   ), invalid_address );
	REQUIRE_THROWS_WITH( s1.dial("tls+tcp://127.0.0.1.32"), invalid_address );
	REQUIRE_THROWS_WITH( s1.dial("tls+tcp://127.0.x.1.32"), invalid_address );
	REQUIRE_THROWS_WITH( s1.dial("tls+tcp://127.0.0.1.32"), invalid_address );
	REQUIRE_THROWS_WITH( s1.dial("tls+tcp://127.0.x.1.32"), invalid_address );
}

TEST_CASE("No delay option", "[tls]" ) {
	REQUIRE_NOTHROW(nng::tls::register_transport());

	auto s = nng::pair::open();
	REQUIRE(s);

	REQUIRE( nng::tcp::get_opt_no_delay(s) );
		
	{
		auto d = nng::dialer( s, "tls+tcp://127.0.0.1:4999" );
		REQUIRE( nng::tcp::get_opt_no_delay(d) );

		REQUIRE_NOTHROW( nng::tcp::set_opt_no_delay(d,false) );
		REQUIRE_FALSE( nng::tcp::get_opt_no_delay(d) );

		auto bad_type = nng::exception(nng::error::badtype).what();

		REQUIRE_THROWS_WITH( d.get_opt_int( NNG_OPT_TCP_NODELAY ), bad_type );
		REQUIRE_THROWS_WITH( d.set_opt_int( NNG_OPT_TCP_NODELAY, 0 ), bad_type );
		
		auto inval = nng::exception(nng::error::inval).what();

		int x = 0;
		REQUIRE_THROWS_WITH( d.set_opt( NNG_OPT_TCP_NODELAY, {&x, sizeof(x)} ), inval );

		nng::listener l( s, "tls+tcp://127.0.0.1:4999" );
		REQUIRE( nng::tcp::get_opt_no_delay(l) );
		
		REQUIRE_THROWS_WITH( l.set_opt_int( NNG_OPT_TCP_NODELAY, 0 ), bad_type );

		x = 0;
		REQUIRE_THROWS_WITH( l.set_opt( NNG_OPT_TCP_NODELAY, {&x, sizeof(x)} ), inval );
	}

	// Make sure socket wide defaults apply.
	REQUIRE_NOTHROW( nng::tcp::set_opt_no_delay(s,true) );
	REQUIRE( nng::tcp::get_opt_no_delay(s) );
		
	REQUIRE_NOTHROW( nng::tcp::set_opt_no_delay(s,false) );

	auto d = nng::dialer( s, "tls+tcp://127.0.0.1:4999" );
	REQUIRE_FALSE( nng::tcp::get_opt_no_delay(d) );
}

TEST_CASE("Keepalive option", "[tls]" ) {
	REQUIRE_NOTHROW(nng::tls::register_transport());

	auto s = nng::pair::open();
	REQUIRE(s);

	REQUIRE_FALSE( nng::tcp::get_opt_keep_alive(s) );
	{
		auto d = nng::dialer( s, "tls+tcp://127.0.0.1:4999" );
		REQUIRE_FALSE( nng::tcp::get_opt_keep_alive(d) );

		REQUIRE_NOTHROW( nng::tcp::set_opt_keep_alive(d,true) );
		REQUIRE( nng::tcp::get_opt_keep_alive(d) );

		auto bad_type = nng::exception(nng::error::badtype).what();

		REQUIRE_THROWS_WITH( d.get_opt_int( NNG_OPT_TCP_KEEPALIVE ), bad_type );
		REQUIRE_THROWS_WITH( d.set_opt_int( NNG_OPT_TCP_KEEPALIVE, 1 ), bad_type );

		auto l = nng::listener( s, "tls+tcp://127.0.0.1:4999" );
		REQUIRE_FALSE( nng::tcp::get_opt_keep_alive(l) );
		
		REQUIRE_THROWS_WITH( l.set_opt_int( NNG_OPT_TCP_KEEPALIVE, 1 ), bad_type );
	}

	// Make sure socket wide defaults apply.
	REQUIRE_NOTHROW( nng::tcp::set_opt_keep_alive(s,false) );
	REQUIRE_FALSE( nng::tcp::get_opt_keep_alive(s) );

	REQUIRE_NOTHROW( nng::tcp::set_opt_keep_alive(s,true) );
	auto d = nng::dialer( s, "tls+tcp://127.0.0.1:4999" );
	REQUIRE( nng::tcp::get_opt_keep_alive(d) );
}
