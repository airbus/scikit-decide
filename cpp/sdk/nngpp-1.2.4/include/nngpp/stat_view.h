#ifndef NNGPP_STAT_VIEW_H
#define NNGPP_STAT_VIEW_H
#include <nng/nng.h>

namespace nng {

enum class stat_type {
	scope   = NNG_STAT_SCOPE,
	level   = NNG_STAT_LEVEL,
	counter = NNG_STAT_COUNTER,
	string  = NNG_STAT_STRING,
	boolean = NNG_STAT_BOOLEAN,
	id      = NNG_STAT_ID
};

enum class stat_unit {
	none     = NNG_UNIT_NONE,
	bytes    = NNG_UNIT_BYTES,
	messages = NNG_UNIT_MESSAGES,
	millis   = NNG_UNIT_MILLIS,
	events   = NNG_UNIT_EVENTS
};

struct stat_view {
protected:
	nng_stat* s = nullptr;
	
public:
	stat_view() = default;
	
	stat_view( nng_stat* s ) noexcept : s(s) {}
	
	nng_stat* get() const noexcept {
		return s;
	}

	nng_stat* operator->() const noexcept {
		return s;
	}

	explicit operator bool() const noexcept {
		return s != nullptr;
	}
	
	void dump() const noexcept {
		nng_stats_dump(s);
	}
	
	stat_view next() const noexcept {
		return nng_stat_next(s);
	}
	
	stat_view child() const noexcept {
		return nng_stat_child(s);
	}

	const char* name() const noexcept {
		return nng_stat_name(s);
	}

	stat_type type() const noexcept {
		return (stat_type)nng_stat_type(s);
	}

	stat_unit unit() const noexcept {
		return (stat_unit)nng_stat_unit(s);
	}

	uint64_t value() const noexcept {
		return nng_stat_value(s);
	}

	const char* string() const noexcept {
		return nng_stat_string(s);
	}
	
	const char* desc() const noexcept {
		return nng_stat_desc(s);
	}
	
	uint64_t timestamp() const noexcept {
		return nng_stat_timestamp(s);
	}
};

}

#endif
