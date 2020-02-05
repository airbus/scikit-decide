#ifndef NNGPP_SURVEY0_H
#define NNGPP_SURVEY0_H
#include <nngpp/ctx.h>
#include <nng/protocol/survey0/survey.h>

namespace nng { namespace survey {
inline namespace v0 {

inline socket open() {
	nng_socket s;
	int r = nng_surveyor0_open(&s);
	if( r != 0 ) {
		throw exception(r,"nng_surveyor0_open");
	}
	return socket(s);
}

inline socket open_raw() {
	nng_socket s;
	int r = nng_surveyor0_open_raw(&s);
	if( r != 0 ) {
		throw exception(r,"nng_surveyor0_open_raw");
	}
	return socket(s);
}

enum class option {
	survey_time // ms
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_SURVEYOR_SURVEYTIME
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline nng_duration get_opt_survey_time( socket_view s ) {
	return s.get_opt_ms( to_name(option::survey_time) );
}

inline void set_opt_survey_time( socket_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::survey_time), v );
}

inline nng_duration get_opt_survey_time( ctx_view s ) {
	return s.get_opt_ms( to_name(option::survey_time) );
}

inline void set_opt_survey_time( ctx_view s, nng_duration v ) {
	s.set_opt_ms( to_name(option::survey_time), v );
}

}
}}

#endif
