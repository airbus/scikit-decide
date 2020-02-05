//
// Copyright 2019 Staysail Systems, Inc. <info@staysail.tech>
//
// This software is supplied under the terms of the MIT License, a
// copy of which should be located in the distribution where this
// file was obtained (LICENSE.txt).  A copy of the license may also be
// found online at https://opensource.org/licenses/MIT.
//

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include <stdbool.h>
#include <stdint.h>

// The following headers are provided for test code convenience.
#include <nng/nng.h>

#ifdef __cplusplus
extern "C" {
#endif

// testutil_clock returns the current time in milliseconds.
// The reference clock may be any point in the past (typically since
// the program started running.)
extern uint64_t testutil_clock(void);

// testutil_pollfd tests if the given file descriptor polls as readable.
extern bool testutil_pollfd(int);

// testutil_htons is just htons portably.
extern uint16_t testutil_htons(uint16_t);

// testutil_htonl is just htonl portably.
extern uint32_t testutil_htonl(uint32_t);

// testutil_sleep sleeps the specified number of msec
extern void testutil_sleep(int);

// testutil_next_port returns a new port number (presumably unique)
extern uint16_t testutil_next_port(void);

// testutil_marry connects two sockets using inproc.  It uses socket
// pipe hooks to ensure that it does not return before both sockets
// are fully connected.
extern int testutil_marry(nng_socket, nng_socket);

// TEST_NNG_PASS tests for NNG success.  It reports the failure if it
// did not.
#define TEST_NNG_PASS(cond)                                          \
	do {                                                         \
		int result_ = (cond);                                \
		TEST_CHECK_(result_ == 0, "%s succeeds", #cond);                    \
		TEST_MSG("%s: expected success, got %s (%d)", #cond, \
		    nng_strerror(result_), result_);                 \
	} while (0)

#define TEST_NNG_FAIL(cond, expect)                                       \
	do {                                                              \
		int result_ = (cond);                                     \
		TEST_CHECK_(result_ == expect, "%s fails with %s", #cond, \
		    nng_strerror(expect));                                \
		TEST_MSG("%s: expected %s (%d), got %s (%d)", #cond,      \
		    nng_strerror(expect), expect, nng_strerror(result_),  \
		    result_);                                             \
	} while (0)

#ifdef __cplusplus
};
#endif

#endif // TESTUTIL_H
