//
// Copyright 2019 Staysail Systems, Inc. <info@staysail.tech>
// Copyright 2018 Capitar IT Group BV <info@capitar.com>
//
// This software is supplied under the terms of the MIT License, a
// copy of which should be located in the distribution where this
// file was obtained (LICENSE.txt).  A copy of the license may also be
// found online at https://opensource.org/licenses/MIT.
//

#include "testutil.h"

#include <nng/nng.h>
#include <nng/supplemental/util/platform.h>

#include "acutest.h"

struct addarg {
	int      cnt;
	nng_mtx *mx;
	nng_cv * cv;
};

void
add(void *arg)
{
	struct addarg *aa = arg;

	nng_mtx_lock(aa->mx);
	aa->cnt++;
	nng_cv_wake(aa->cv);
	nng_mtx_unlock(aa->mx);
}

void
test_sleep(void)
{
	uint64_t start, end;
	start = testutil_clock();
	nng_msleep(100);
	end = testutil_clock();
	TEST_CHECK((end - start) >= 100);
	TEST_CHECK((end - start) <= 500);
}

void
test_clock(void)
{
	uint64_t mstart;
	uint64_t msend;
	uint64_t usdelta;
	uint64_t msdelta;
	nng_time usend;
	nng_time usnow;

	mstart = testutil_clock();
	usnow  = nng_clock();
	nng_msleep(200);
	usend = nng_clock();
	msend = testutil_clock();

	TEST_CHECK(usend > usnow);
	TEST_CHECK(msend > mstart);
	usdelta = usend - usnow;
	msdelta = msend - mstart;
	TEST_CHECK(usdelta >= 200);
	TEST_CHECK(usdelta < 500); // increased tolerance for CIs
	if (msdelta > usdelta) {
                TEST_CHECK((msdelta - usdelta) < 50);
	} else {
                TEST_CHECK((usdelta - msdelta) < 50);
        }
}

void
test_mutex(void)
{
	nng_mtx *mx, *mx2;

	TEST_CHECK(nng_mtx_alloc(&mx) == 0);
	nng_mtx_lock(mx);
	nng_mtx_unlock(mx);

	nng_mtx_lock(mx);
	nng_mtx_unlock(mx);
	nng_mtx_free(mx);

	// Verify that the mutexes are not always the same!
	TEST_CHECK(nng_mtx_alloc(&mx) == 0);
	TEST_CHECK(nng_mtx_alloc(&mx2) == 0);
	TEST_CHECK(mx != mx2);
	nng_mtx_free(mx);
	nng_mtx_free(mx2);
}

void
test_thread(void)
{
	nng_thread *  thr;
	int           rv;
	struct addarg aa;

	TEST_CHECK(nng_mtx_alloc(&aa.mx) == 0);
	TEST_CHECK(nng_cv_alloc(&aa.cv, aa.mx) == 0);
	aa.cnt = 0;

	TEST_CHECK((rv = nng_thread_create(&thr, add, &aa)) == 0);
	nng_thread_destroy(thr);
	TEST_CHECK(aa.cnt == 1);

        nng_cv_free(aa.cv);
	nng_mtx_free(aa.mx);
}

void
test_condvar(void)
{
        nng_thread *  thr;
        int           rv;
        struct addarg aa;

        TEST_CHECK(nng_mtx_alloc(&aa.mx) == 0);
        TEST_CHECK(nng_cv_alloc(&aa.cv, aa.mx) == 0);
        aa.cnt = 0;

        TEST_CHECK((rv = nng_thread_create(&thr, add, &aa)) == 0);

        nng_mtx_lock(aa.mx);
        while (aa.cnt == 0) {
                nng_cv_wait(aa.cv);
        }
        nng_mtx_unlock(aa.mx);
        nng_thread_destroy(thr);
        TEST_CHECK(aa.cnt == 1);

        nng_cv_free(aa.cv);
        nng_mtx_free(aa.mx);
}

TEST_LIST = {
	{ "sleep", test_sleep },
	{ "clock", test_clock },
	{ "mutex", test_mutex },
        { "thread", test_thread },
        { "condvar", test_condvar },
	{ NULL, NULL },
};
