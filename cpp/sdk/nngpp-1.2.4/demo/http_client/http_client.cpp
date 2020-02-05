// This is a port of the nng demo to nngpp
// See https://github.com/nanomsg/nng/tree/master/demo/http_client

#include <cstdio>
#include <cstdlib>
#include <nngpp/nngpp.h>
#include <nngpp/http/http.h>

int main( int argc, char** argv ) try {
	if(argc < 2) {
		fprintf(stderr, "No URL supplied!\n");
		return 1;
	}

	nng::url url( argv[1] );
	nng::http::client client( url );

	auto aio = nng::make_aio();
	client.connect( aio );

	aio.wait();
	{
		auto result = aio.result();
		if( result != nng::error::success ) {
			throw nng::exception(result);
		}
	}

	// Get the connection, at the 0th output.
	nng::http::conn_view conn = aio.get_output<nng_http_conn>(0);

	// Request is already set up with URL, and for GET via HTTP/1.1.
	// The Host: header is already set up too.

	// Send the request, and wait for that to finish.
	nng::http::req req(url);
	conn.write(req,aio);
	aio.wait();
	{
		auto result = aio.result();
		if( result != nng::error::success ) {
			throw nng::exception(result);
		}
	}
	
	// Read a response.
	auto res = nng::http::make_res();
	conn.read(res,aio);
	aio.wait();
	{
		auto result = aio.result();
		if( result != nng::error::success ) {
			throw nng::exception(result);
		}
	}

	if(res.get_status() != nng::http::status::ok) {
		fprintf(stderr, "HTTP Server Responded: %d %s\n", (int)res.get_status(), res.get_reason());
	}

	// This only supports regular transfer encoding (no Chunked-Encoding,
	// and a Content-Length header is required.)
	auto hdr = res.get_header("Content-Length");
	if(!hdr) {
		fprintf(stderr, "Missing Content-Length header.\n");
		return 1;
	}

	auto len = atoi(hdr);
	if(len == 0) {
		return 0;
	}

	// Allocate a buffer to receive the body data.
	nng::buffer data(len);

	// Set up a single iov to point to the buffer.
	// Following never fails with fewer than 5 elements.
	aio.set_iov( data );

	// Now attempt to receive the data.
	conn.read_all(aio);

	// Wait for it to complete.
	aio.wait();
	{
		auto result = aio.result();
		if( result != nng::error::success ) {
			throw nng::exception(result);
		}
	}

	fwrite(data.data(), 1, data.size(), stdout);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
