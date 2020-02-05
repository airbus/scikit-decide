#include <catch2/catch.hpp>
#include <nngpp/nngpp.h>
#include <nngpp/protocol/pull0.h>
#include <nngpp/protocol/push0.h>
#include <nngpp/platform/platform.h>
#include <mutex>

namespace {

struct testcase {
	nng::mtx lk = nng::make_mtx();
	nng::cv cv{ lk };
	nng::socket s;
	nng::dialer d;
	nng::listener l;
	nng::pipe_view p;
	int add_pre = 0;
	int add_post = 0;
	int rem = 0;
	int err = 0;
	int rej = 0;

	testcase( nng::socket&& s ) : s(std::move(s)) {}
};

bool expect(testcase* t, int* vp, int v) {
	nng_time when = nng_clock() + 5000; // five seconds
	
	std::lock_guard<nng::mtx> lock( t->lk );
	while(*vp != v) {
		if( !t->cv.wait_until(when) ) {
			break;
		}
	}
	bool ok = (*vp == v);
	if(!ok) {
		printf("Expected %d but got %d\n", v, *vp);
	}
	return ok;
}

void notify(nng_pipe pp, nng_pipe_ev act, void* arg) {
	 auto t = (testcase*)arg;
	 nng::pipe_view p = pp;
	 
	std::lock_guard<nng::mtx> lock( t->lk );
	if(
		(p.get_socket().id() != t->s.id()) ||
	    (p.get_listener().id() != t->l.id()) ||
	    (p.get_dialer().id() != t->d.id())
	) {
		t->err++;
		t->cv.wake_all();
		return;
	}
	if (t->add_post > t->add_pre) {
		t->err++;
	}
	switch(act) {
	case NNG_PIPE_EV_ADD_PRE:
		t->add_pre++;
		break;
	case NNG_PIPE_EV_ADD_POST:
		t->add_post++;
		break;
	case NNG_PIPE_EV_REM_POST:
		t->rem++;
		break;
	default:
		t->err++;
		t->cv.wake_all();
		return;
	}
	t->p = p;
	t->cv.wake_all();
}

void reject(nng_pipe pp, nng_pipe_ev act, void* arg) {
	auto t = (testcase*)arg;
	notify(pp, act, arg);
	
	std::lock_guard<nng::mtx> lock( t->lk );
	if(!t->rej) {
		{
			nng::pipe pipe(pp);
		}
		t->rej++;
	}
}

}

TEST_CASE("Pipe notify works","[pipe]") {

	SECTION("We can create a pipeline") {
		testcase push = nng::push::open();
		testcase pull = nng::pull::open();
		auto addr = "inproc://pipe";

		REQUIRE_NOTHROW( set_opt_reconnect_time_min(push.s,10) );
		REQUIRE_NOTHROW( set_opt_reconnect_time_max(push.s,10) );
		REQUIRE_NOTHROW( set_opt_reconnect_time_min(pull.s,10) );
		REQUIRE_NOTHROW( set_opt_reconnect_time_max(pull.s,10) );

		REQUIRE_NOTHROW( push.s.pipe_notify( nng::pipe_ev::add_pre, notify, &push ) );
		REQUIRE_NOTHROW( push.s.pipe_notify( nng::pipe_ev::add_post, notify, &push ) );
		REQUIRE_NOTHROW( push.s.pipe_notify( nng::pipe_ev::rem_post, notify, &push ) );
		REQUIRE_NOTHROW( pull.s.pipe_notify( nng::pipe_ev::add_pre, notify, &pull ) );
		REQUIRE_NOTHROW( pull.s.pipe_notify( nng::pipe_ev::add_post, notify, &pull ) );
		REQUIRE_NOTHROW( pull.s.pipe_notify( nng::pipe_ev::rem_post, notify, &pull ) );

		SECTION("Dialing works") {
			REQUIRE_NOTHROW( pull.l = nng::listener(pull.s,addr) );
			REQUIRE_NOTHROW( push.d = nng::dialer(push.s,addr) );
			
			REQUIRE(pull.l);
			REQUIRE(push.d);

			REQUIRE_NOTHROW( set_opt_reconnect_time_min(push.d,10) );
			REQUIRE_NOTHROW( set_opt_reconnect_time_max(push.d,10) );

			REQUIRE_NOTHROW( pull.l.start() );
			REQUIRE_NOTHROW( push.d.start() );

			REQUIRE(expect(&pull, &pull.add_pre, 1));
			REQUIRE(expect(&pull, &pull.add_post, 1));
			REQUIRE(expect(&pull, &pull.add_pre, 1));
			REQUIRE(expect(&pull, &pull.add_post, 1));
			REQUIRE(expect(&pull, &pull.rem, 0));
			REQUIRE(expect(&pull, &pull.err, 0));
			REQUIRE(expect(&push, &push.add_pre, 1));
			REQUIRE(expect(&push, &push.add_post, 1));
			REQUIRE(expect(&push, &push.rem, 0));
			REQUIRE(expect(&push, &push.err, 0));

			SECTION("We can send a frame") {
				auto msg = nng::make_msg(0);
				msg.body().append("hello");
				push.s.send( std::move(msg) );
				msg = pull.s.recv_msg();
				REQUIRE(msg);
				REQUIRE( msg.body().get() == "hello" );
				REQUIRE( msg.get_pipe().id() == pull.p.id() );
			}

			SECTION("Reconnection works") {
				REQUIRE(expect(&pull, &pull.add_pre, 1));
				REQUIRE(expect(&pull, &pull.add_post, 1));
				nng_pipe_close(pull.p.get());

				REQUIRE(expect(&pull, &pull.rem, 1));
				REQUIRE(expect(&pull, &pull.err, 0));
				REQUIRE(expect(&pull, &pull.add_pre, 2));
				REQUIRE(expect(&pull, &pull.add_post, 2));

				REQUIRE(expect(&push, &push.rem, 1));
				REQUIRE(expect(&push, &push.err, 0));
				REQUIRE(expect(&push, &push.add_pre, 2));
				REQUIRE(expect(&push, &push.add_post, 2));

				SECTION("They still exchange frames") {
					nng::msleep(200);
					nng::pipe_view p1;
					{
						auto msg = nng::make_msg(0);
						msg.body().append("hello");
						push.s.send( std::move(msg) );
						msg = pull.s.recv_msg();
						REQUIRE(msg);
						REQUIRE( msg.body().get() == "hello" );
						REQUIRE( msg.get_pipe().id() == pull.p.id() );

						p1 = msg.get_pipe();
					}
					REQUIRE(p1.id() == pull.p.id());
				}
			}
		}

		SECTION("Reject works") {
			
			REQUIRE_NOTHROW( pull.s.pipe_notify( nng::pipe_ev::add_pre, reject, &pull ) );

			REQUIRE_NOTHROW( pull.l = nng::listener(pull.s,addr) );
			REQUIRE_NOTHROW( push.d = nng::dialer(push.s,addr) );
			
			REQUIRE(pull.l);
			REQUIRE(push.d);
			
			REQUIRE_NOTHROW( pull.l.start() );
			nng::msleep(100);
			REQUIRE_NOTHROW( push.d.start() );
			
			REQUIRE(expect(&pull, &pull.add_pre, 2));
			REQUIRE(expect(&pull, &pull.add_post, 1));
			REQUIRE(expect(&pull, &pull.rem, 1));
			REQUIRE(expect(&pull, &pull.err, 0));
			REQUIRE(expect(&push, &push.add_pre, 2));
			REQUIRE(expect(&push, &push.add_post, 2));
			REQUIRE(expect(&push, &push.rem, 1));
			REQUIRE(expect(&push, &push.err, 0));
		}
	}
}
