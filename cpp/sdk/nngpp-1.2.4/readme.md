# nngpp

C++ wrapper around the [nanomsg NNG](https://github.com/nanomsg/nng) API

What's nanomsg? In a nutshell it's a spiritual successor to ZeroMQ, and NNG is its latest incarnation.

## Features

* Header-only -- just add the `include` directory to your includes
* Supports all nng protocols, transports and supplemental code
* Zero overhead -- no virtual functions, no extra allocations or copies
* Owning RAII classes, e.g. `nng::socket` and `nng::msg`
* Non-owning views, e.g. `nng::socket_view` and `nng::msg_view`
* Failure is communicated with exceptions instead of error codes
* Compatible with C++11 but uses C++17 features when available

## Hello world

```cpp
#include <nngpp/nngpp.h>
#include <nngpp/protocol/req0.h>
#include <nngpp/protocol/rep0.h>
#include <cstdio>

int main() try {
	// create a socket for the rep protocol
	nng::socket rep_sock = nng::rep::open();
	
	// rep starts listening using the tcp transport
	rep_sock.listen( "tcp://localhost:8000" );
	
	// create a socket for the req protocol
	nng::socket req_sock = nng::req::open();
	
	// req dials and establishes a connection
	req_sock.dial( "tcp://localhost:8000" );
	
	// req sends "hello" including the null terminator
	req_sock.send("hello");
	
	// rep receives a message
	nng::buffer rep_buf = rep_sock.recv();
	
	// check the content
	if( rep_buf == "hello" ) {
		// rep sends "world" in response
		rep_sock.send("world");
	}
	
	// req receives "world"
	nng::buffer req_buf = req_sock.recv();
}
catch( const nng::exception& e ) {
	// who() is the name of the nng function that produced the error
	// what() is a description of the error code
	printf( "%s: %s\n", e.who(), e.what() );
	return 1;
}
// req_buf is freed
// rep_buf is freed
// req_sock is closed
// rep_sock is closed
```

## Demos

All 5 [nng demos](https://github.com/nanomsg/nng/tree/master/demo) have been ported to nngpp to illustrate the use of the library.
I have kept the original structure of the demos intact, just replacing the nng API with nngpp.
This allows for easy comparison with the nng demos, but may mean they are non-idiomatic in places.
See the [`demo`](demo) directory.


## Dependencies

Since nngpp is a wrapper around nng, you will need to link with libnng.

* If you built nng without tls support:
`-lnng -lpthread`
* If you built nng with tls support:
`-lnng -lmbedtls -lmbedx509 -lmbedcrypto -lpthread`


## Design

The nng API consists of a number of conceptual objects, such as 'socket' and 'pipe'.
Each of these is identified and referred to by a handle, which is typically either a pointer or a (structured) integer id.

In nngpp these handles are wrapped in *views* that expose all of the corresponding functionality as member functions.
For example, the `nng_socket` handle is wrapped by a `nng::socket_view` that has the member function `dial()`, which calls the nng function `nng_dial()`.
All views provide a common set of member functions:
* default constructor of null handle
* implicit construction from a given handle
* `get()` -- get the handle
* `operator->()` (for pointer handles) -- access to the handle's members
* `operator bool()` -- test for the presence of a non-null handle

Most of these conceptual objects are also *resources*, i.e. they must be acquired and released.
To manage the ownership of these resources, nngpp provides RAII classes, such as `nng::socket`.
Each RAII class is an extension of the corresponding view and has the following:
* default constructor of null handle
* explicit construction from a given handle
* specific constructors -- acquire a new handle using the given arguments
* destructor -- release the resource (if held)
* move -- transfer ownership to another instance
* copy (if possible) -- duplicate the resource
* `release()` -- get the underlying handle, detaching its ownership from the object

In addition to constructors, there are make functions:
```cpp
auto msg1 = nng::make_msg(32);  // acquire using make function
 
nng::msg msg2(32);              // acquire using constructor
```

The make function is needed to acquire with zero arguments, e.g.
```cpp
auto mtx1 = nng::make_mtx();  // acquire new mutex
 
nng::mtx mtx2;                // default constructor gives null mutex
```

### Summary

| Name               | nng handle             | nngpp view                   | nngpp RAII              |
|:-------------------|:-----------------------|:-----------------------------|:------------------------|
| socket             | `nng_socket`           | `nng::socket_view`           | `nng::socket`           |
| context            | `nng_ctx`              | `nng::ctx_view`              | `nng::ctx`              |
| dialer             | `nng_dialer`           | `nng::dialer_view`           | `nng::dialer`           |
| listener           | `nng_listener`         | `nng::listener_view`         | `nng::listener`         |
| pipe               | `nng_pipe`             | `nng::pipe_view`             | `nng::pipe`             |
| message            | `nng_msg*`             | `nng::msg_view`              | `nng::msg`              |
| async i/o          | `nng_aio*`             | `nng::aio_view`              | `nng::aio`              |
| url                | `nng_url*`             | `nng::url_view`              | `nng::url`              |
| buffer             | `void*, size_t`        | `nng::view`                  | `nng::buffer`           |
| stat               | `nng_stat*`            | `nng::stat_view`             | `nng::stat`             |
| thread             | `nng_thread*`          | `nng::thread_view`           | `nng::thread`           |
| mutex              | `nng_mtx*`             | `nng::mtx_view`              | `nng::mtx`              |
| condition variable | `nng_cv*`              | `nng::cv_view`               | `nng::cv`               |
| tls config         | `nng_tls_config*`      | `nng::tls::config_view`      | `nng::tls::config`      |
| http client        | `nng_http_client*`     | `nng::http::client_view`     | `nng::http::client`     |
| http connection    | `nng_http_conn*`       | `nng::http::conn_view`       | `nng::http::conn`       |
| http handler       | `nng_http_handler*`    | `nng::http::handler_view`    | `nng::http::handler`    |
| http request       | `nng_http_req*`        | `nng::http::req_view`        | `nng::http::req`        |
| http response      | `nng_http_res*`        | `nng::http::res_view`        | `nng::http::res`        |
| http server        | `nng_http_server*`     | `nng::http::server_view`     | `nng::http::server`     |
| stream             | `nng_stream*`          | `nng::stream::stream_view`   | `nng::stream::stream`   |
| stream dialer      | `nng_stream_dialer*`   | `nng::stream::dialer_view`   | `nng::stream::dialer`   |
| stream listener    | `nng_stream_listener*` | `nng::stream::listener_view` | `nng::stream::listener` |


### Error handling

Many of nng's functions can fail and do so by returning an error code.
In nngpp, the wrapper will throw these error codes as an `nng::exception` that has:
* `get_error()` -- get the error code
* `who()` -- name of the nng function that failed
* `what()` -- description of the error

All nngpp functions that don't throw are marked `noexcept`.

### Failure in destructors

The following objects have destructors that can fail:
* `nng::socket`
* `nng::ctx`
* `nng::listener`
* `nng::dialer`
* `nng::pipe`

In this case the destructor completes without throwing.


### Protocol versioning

NNG has version numbers on its protocols, e.g. pair0 and pair1.
This is expressed in nngpp with namespaces:

```cpp
auto s0 = nng::pair::v0::open();  // version 0

auto s1 = nng::pair::v1::open();  // version 1

auto s2 = nng::pair::open();      // latest version (currently v1)
```

An inline namespace is used for the latest version of each protocol.

### Socket options

Some objects (socket, ctx, dialer, listener, pipe) have options that can be get/set.
The nng API requires that you use the correct type of getter/setter function for each option.
For example, the receive timeout socket option (`NNG_OPT_RECVTIMEO`) is of type `nng_duration` and must set with `nng_setopt_ms()`.
Another issue is that some options are read-only.
In both cases you will only find out at run-time if there is a problem with your code.

In nngpp every option on every object has a dedicated getter and setter of the correct type:
```cpp
nng::set_opt_recv_timeout( socket, 1000 );
nng_duration timeout = nng::get_opt_recv_timeout( socket );
```

If an option is read-only it will have no setter -- you will know at compile time or earlier if you are trying to set a read-only option.

These dedicated getters and setters are free-standing functions, rather than members, as there are some options that are only available with certain transports.
For example, with the TCP transport the socket has a keep alive option:
```cpp
bool keep_alive = nng::tcp::get_opt_keep_alive( socket );
```
