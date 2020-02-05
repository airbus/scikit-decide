#ifndef NNGPP_PERF_LATENCY_H
#define NNGPP_PERF_LATENCY_H
#include <nngpp/nngpp.h>
#include <nngpp/platform/platform.h>
#include <nngpp/protocol/pair1.h>
#include "misc.h"

void latency_client(const char* addr, size_t msgsize, int trips) {
	nng_time start, end;
	{
		auto s = nng::pair::open();
		s.dial(addr);
		nng::msg msg(msgsize);

		start = nng::clock();
		for(int i = 0; i < trips; ++i) {
			s.send( std::move(msg) );
			msg = s.recv_msg();
		}
		end = nng::clock();
	}

	auto total = (float)(end - start) / 1000;
	auto latency = (float)(total * 1000000) / (trips * 2);
	printf("total time: %.3f [s]\n", total);
	printf("message size: %d [B]\n", (int)msgsize);
	printf("round trip count: %d\n", trips);
	printf("average latency: %.3f [us]\n", latency);
}

void latency_server(const char* addr, size_t msgsize, int trips) {
	auto s = nng::pair::open();
	s.listen(addr);
	for(int i = 0; i < trips; ++i) {
		auto msg = s.recv_msg();
		if( msg.body().size() != msgsize ) {
			die("wrong message size: %d != %d", msg.body().size(), msgsize);
		}
		s.send( std::move(msg) );
	}

	// Wait a bit for things to drain... linger should do this.
	// 100ms ought to be enough.
	nng::msleep(100);
}

#endif
