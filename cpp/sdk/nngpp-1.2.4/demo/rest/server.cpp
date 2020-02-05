// This is a port of the nng demo to nngpp
// See https://github.com/nanomsg/nng/tree/master/demo/rest

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/req0.h>
#include <nngpp/protocol/rep0.h>
#include <nngpp/http/http.h>
#include <nngpp/platform/platform.h>

#define INPROC_URL "inproc://rot13"
#define REST_URL "http://127.0.0.1:%u/api/rest/rot13"

// This server acts as a proxy.  We take HTTP POST requests, convert them to
// REQ messages, and when the reply is received, send the reply back to
// the original HTTP client.
//
// The state flow looks like:
//
// 1. Receive HTTP request & headers
// 2. Receive HTTP request (POST) data
// 3. Send POST payload as REQ body
// 4. Receive REP reply (including payload)
// 5. Return REP message body to the HTTP server (which forwards to client)
// 6. Restart at step 1.
//
// The above flow is pretty linear, and so we use contexts (nng_ctx) to
// obtain parallelism.

enum job_state {
	SEND_REQ,  // Sending REQ request
	RECV_REP   // Receiving REQ reply
};

static void rest_job_cb(void* arg);

struct rest_job {
	nng::aio_view             http_aio;                 // aio from HTTP we must reply to
	nng::http::res            http_res;                 // HTTP response object
	job_state                 state = SEND_REQ;         // 0 = sending, 1 = receiving
	nng::msg                  msg;                      // request message
	nng::aio                  aio{ rest_job_cb, this }; // request flow
	nng::ctx                  ctx;                      // context on the request socket
	std::unique_ptr<rest_job> next;                     // next on the freelist
};

nng::socket req_sock;

// We maintain a queue of free jobs.  This way we don't have to
// deallocate them from the callback; we just reuse them.
auto job_lock = nng::make_mtx();
std::unique_ptr<rest_job> job_freelist;

static void rest_recycle_job(std::unique_ptr<rest_job>&& job) {
	job->http_res = nng::http::res();
	job->msg = nng::msg();
	job->ctx = nng::ctx();

	std::lock_guard<nng::mtx> lock( job_lock );
	job->next    = std::move(job_freelist);
	job_freelist = std::move(job);
}

static std::unique_ptr<rest_job> rest_get_job(void) {
	std::unique_ptr<rest_job> job;
	{
		job_lock.lock();
		job = std::move(job_freelist);
		if( job ) {
			job_freelist = std::move(job->next);
			job_lock.unlock();
			return job;
		}
		job_lock.unlock();
	}

	job = std::make_unique<rest_job>();
	if( !job ) {
		return {};
	}
	return job;
}

static void rest_http_fatal(std::unique_ptr<rest_job>&& job, const char* who, nng::error what) {
	
	nng::aio_view aio = job->http_aio;
	job->http_aio = nullptr;
	nng::http::res res = std::move(job->http_res);

	char buf[128];
	snprintf(buf, sizeof(buf), "%s: %s", who, nng::to_string(what));

	res.set_status( nng::http::status::internal_server_error );
	res.set_reason( buf );
	aio.set_output(0,res.release());
	aio.finish();
	rest_recycle_job(std::move(job));
}

static void rest_job_cb(void* arg) {
	std::unique_ptr<rest_job> job( (rest_job*)arg );
	nng::aio_view aio = job->aio;

	switch(job->state) {
	case SEND_REQ:
		{
			auto result = aio.result();
			if( result != nng::error::success ) {
				rest_http_fatal(std::move(job), "send REQ failed", result);
				return;
			}
		}
		// Message was sent, so now wait for the reply.
		aio.set_msg(nullptr);
		job->state = RECV_REP;
		job->ctx.recv( aio );
		job.release();
		break;
	case RECV_REP:
		{
			auto result = aio.result();
			if( result != nng::error::success ) {
				rest_http_fatal(std::move(job), "recv reply failed", result);
				return;
			}
		}
		job->msg = aio.release_msg();
		// We got a reply, so give it back to the server.
		try {
			job->http_res.copy_data( job->msg.body().get() );
		}
		catch( const nng::exception& e ) {
			rest_http_fatal(std::move(job), e.who(), e.get_error());
			return;
		}
		// Set the output - the HTTP server will send it back to the
		// user agent with a 200 response.
		job->http_aio.set_output(0, job->http_res.release());
		job->http_aio.finish();
		job->http_aio = nullptr;
		// We are done with the job.
		rest_recycle_job(std::move(job));
		return;
	default:
		throw nng::exception(nng::error::state,"bad case");
		break;
	}
}

// Our rest server just takes the message body, creates a request ID
// for it, and sends it on.  This runs in raw mode, so
void rest_handle(nng_aio* a) {
	nng::aio_view aio = a;
	nng::http::req_view req = aio.get_input<nng_http_req>(0);

	auto job = rest_get_job();
	if( !job ) {
		aio.finish( nng::error::nomem );
		return;
	}
	try {
		job->http_res = nng::http::make_res();
		job->ctx = nng::ctx(req_sock);
	}
	catch( const nng::exception& e ) {
		rest_recycle_job( std::move(job) );
		aio.finish( e.get_error() );
		return;
	}

	auto data = req.get_data();
	job->http_aio = aio;

	try {
		job->msg = nng::msg(data.size());
	}
	catch( const nng::exception& e ) {
		rest_http_fatal(std::move(job), e.who(), e.get_error());
		return;
	}

	memcpy(job->msg.body().data(), data.data(), data.size());
	job->aio.set_msg( std::move(job->msg) );
	job->state = SEND_REQ;
	job->ctx.send(job->aio);
	job.release();
}

nng::http::server rest_start(uint16_t port) {

	// Set up some strings, etc.  We use the port number
	// from the argument list.
	char rest_addr[128];
	snprintf(rest_addr, sizeof(rest_addr), REST_URL, port);

	nng::url url(rest_addr);

	// Create the REQ socket, and put it in raw mode, connected to
	// the remote REP server (our inproc server in this case).
	req_sock = nng::req::open();

	req_sock.dial( INPROC_URL, nng::flag::nonblock );

	// Get a suitable HTTP server instance.  This creates one
	// if it doesn't already exist.
	nng::http::server server(url);

	// Allocate the handler - we use a dynamic handler for REST
	// using the function "rest_handle" declared above.
	nng::http::handler handler( url->u_path, rest_handle );
	handler.set_method( nng::http::verb::post );
	// We want to collect the body, and we (arbitrarily) limit this to
	// 128KB.  The default limit is 1MB.  You can explicitly collect
	// the deta yourself with another HTTP read transaction by disabling
	// this, but that's a lot of work, especially if you want to handle
	// chunked transfers.
	handler.collect_body(true, 1024 * 128);
	server.add_handler( std::move(handler) );
	server.start();
	return server;
}

//
// inproc_server - this just is a simple REP server that listens for
// messages, and performs ROT13 on them before sending them.  This
// doesn't have to be in the same process -- it is hear for demonstration
// simplicity only.  (Most likely this would be somewhere else.)  Note
// especially that this uses inproc, so nothing can get to it directly
// from outside the process.
//
void inproc_server(void*) try {

	auto sock = nng::rep::open();
	sock.listen( INPROC_URL );

	// This is simple enough that we don't need concurrency.  Plus it
	// makes for an easier demo.
	while(true) {
		auto msg = sock.recv_msg();
		auto body = msg.body().data<char>();
		auto size = (int)msg.body().size();
		for(int i = 0; i < size; ++i) {
			// Table lookup would be faster, but this works.
			if( isupper(body[i]) ) {
				auto base = body[i] - 'A';
				base = (base + 13) % 26;
				body[i] = base + 'A';
			} else if( islower(body[i]) ) {
				auto base = body[i] - 'a';
				base = (base + 13) % 26;
				body[i] = base + 'a';
			}
		}
		sock.send( std::move(msg) );
	}
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
}

int main(int argc, char** argv) try {
	nng::http::server server;
	nng::thread inproc_thr( inproc_server, nullptr );

	uint16_t port = 0;
	if(getenv("PORT") != nullptr) {
		port = (uint16_t)atoi(getenv("PORT"));
	}
	port = port ? port : 8888;
	server = rest_start(port);
}
catch( const nng::exception& e ) {
	fprintf(stderr, "%s: %s\n", e.who(), e.what());
	return 1;
}
