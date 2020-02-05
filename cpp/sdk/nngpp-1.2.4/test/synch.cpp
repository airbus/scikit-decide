#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/platform/platform.h>
#include <mutex>

namespace {

struct notifyarg {
	int          did = 0;
	nng_duration when = 0;
	nng::mtx     mx = nng::make_mtx();
	nng::cv      cv{mx};
};

static void notifyafter(void* arg) {
	auto na = (notifyarg*)arg;

	nng::msleep(na->when);
	std::lock_guard<nng::mtx> lock( na->mx );
	na->did = 1;
	na->cv.wake_all();
}

}

TEST_CASE("We can lock and unlock a mutex","[synch]") {
	auto mx = nng::make_mtx();
	mx.lock();
	REQUIRE(1);
	mx.unlock();
	REQUIRE(1);
	SECTION("And then lock it again","[synch]") {
		mx.lock();
		REQUIRE(1);
		mx.unlock();
		REQUIRE(1);
	}
}
TEST_CASE("Things block properly","[synch]") {
	notifyarg arg;
	nng::thread thr;
	{
		std::lock_guard<nng::mtx> lock( arg.mx );
		thr = nng::thread(notifyafter,&arg);
		nng::msleep(10);
		REQUIRE(arg.did == 0);
	}
	nng::msleep(10);
	{
		std::lock_guard<nng::mtx> lock( arg.mx );
		while(!arg.did) {
			arg.cv.wait();
		}
		REQUIRE(arg.did != 0);
	}
}

TEST_CASE("Notification works","[synch]") {
	notifyarg arg;
	arg.did  = 0;
	arg.when = 10;
	{
		nng::thread thr(notifyafter,&arg);
		std::lock_guard<nng::mtx> lock( arg.mx );
		if(!arg.did) {
			arg.cv.wait();
		}
	}
	REQUIRE(arg.did == 1);
}

TEST_CASE("Timeout works","[synch]") {
	notifyarg arg;
	arg.did  = 0;
	arg.when = 200;
	nng::thread thr(notifyafter,&arg);
	std::lock_guard<nng::mtx> lock( arg.mx );
	if(!arg.did) {
		arg.cv.wait_until(nng::clock() + 10);
	}
	REQUIRE(arg.did == 0);
}

TEST_CASE("Empty timeout is EAGAIN","[synch]") {
	notifyarg arg;
	std::lock_guard<nng::mtx> lock( arg.mx );
	REQUIRE_THROWS_WITH(
		arg.cv.wait_until(0),
		nng::exception(nng::error::again).what()
	);
}

TEST_CASE("Not running works","[synch]") {
	notifyarg arg;
	arg.did  = 0;
	arg.when = 1;
	std::lock_guard<nng::mtx> lock( arg.mx );
	if(!arg.did) {
		arg.cv.wait_until(nng::clock() + 10);
	}
	REQUIRE(arg.did == 0);
}
