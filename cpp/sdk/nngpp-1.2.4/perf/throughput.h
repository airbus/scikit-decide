#ifndef NNGPP_PERF_THROUGHPUT_H
#define NNGPP_PERF_THROUGHPUT_H
#include <nngpp/nngpp.h>
#include <nngpp/platform/platform.h>
#include <nngpp/protocol/pair1.h>
#include "misc.h"

void throughput_server(const char* addr, size_t msgsize, int count) {
	nng_time start, end;
	{
		auto s = nng::pair::open();
		nng::set_opt_recv_buffer( s, 128 );
		s.listen(addr);

		// Receive first synchronization message.
		s.recv_msg();
		start = nng::clock();

		for(int i = 0; i < count; ++i) {
			auto msg = s.recv_msg();
			if(msg.body().size() != msgsize) {
				die("wrong message size: %d != %d", msg.body().size(), msgsize);
			}
		}
		end = nng::clock();
		// Send a synchronization message (empty) to the other side,
		// and wait a bit to make sure it goes out the wire.
		s.send( nng::view("",0) );
		nng::msleep(200);
	}
	auto total = (float)(end - start) / 1000;
	auto msgpersec = (float)count / total;
	auto mbps = (float)(msgpersec * 8 * msgsize) / (1024 * 1024);
	printf("total time: %.3f [s]\n", total);
	printf("message size: %d [B]\n", (int)msgsize);
	printf("message count: %d\n", count);
	printf("throughput: %.f [msg/s]\n", msgpersec);
	printf("throughput: %.3f [Mb/s]\n", mbps);
}

void throughput_client(const char* addr, size_t msgsize, int count) {
	auto s = nng::pair::open();
	nng::set_opt_send_buffer( s, 128 );
	nng::set_opt_recv_timeout( s, 5000 );
	s.dial(addr);

	// We send one extra zero length message to start the timer.
	s.send( nng::make_msg(0) );

	for(int i = 0; i < count; ++i) {
		s.send( nng::make_msg(msgsize) );
	}
	// Attempt to get the completion indication from the other side.
	s.recv_msg();
}

#endif
