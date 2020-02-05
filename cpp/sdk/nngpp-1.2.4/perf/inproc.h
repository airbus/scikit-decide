#ifndef NNGPP_PERF_INPROC_H
#define NNGPP_PERF_INPROC_H

struct inproc_args {
	int count;
	int msgsize;
	const char* addr;
	void (*func)(const char*, size_t, int);
};

static void do_inproc(void* args) {
	auto ia = (inproc_args*)args;
	ia->func(ia->addr, ia->msgsize, ia->count);
}

#endif
