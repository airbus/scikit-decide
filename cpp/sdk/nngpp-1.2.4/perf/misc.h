#ifndef NNGPP_PERF_MISC_H
#define NNGPP_PERF_MISC_H
#include <cstdlib>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>

static int parse_int(const char* arg, const char* what) {
	char* eptr;
	auto val = strtol(arg, &eptr, 10);
	// Must be a postive number less than around a billion.
	if((val < 0) || (val > (1 << 30)) || (*eptr != 0) || (eptr == arg)) {
		throw std::runtime_error(what);
	}
	return (int)val;
}

static void die(const char* fmt, ...) {
	va_list ap;
	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
	fprintf(stderr, "\n");
	exit(2);
}

#endif
