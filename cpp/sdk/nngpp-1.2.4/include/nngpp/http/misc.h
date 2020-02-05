#ifndef NNGPP_HTTP_MISC_H
#define NNGPP_HTTP_MISC_H
#include <nng/supplemental/http/http.h>

namespace nng { namespace http {

enum class status {
	continue_                = NNG_HTTP_STATUS_CONTINUE,
	switching                = NNG_HTTP_STATUS_SWITCHING,
	processing               = NNG_HTTP_STATUS_PROCESSING,
	ok                       = NNG_HTTP_STATUS_OK,
	created                  = NNG_HTTP_STATUS_CREATED,
	accepted                 = NNG_HTTP_STATUS_ACCEPTED,
	not_authoritative        = NNG_HTTP_STATUS_NOT_AUTHORITATIVE,
	no_content               = NNG_HTTP_STATUS_NO_CONTENT,
	reset_content            = NNG_HTTP_STATUS_RESET_CONTENT,
	partial_content          = NNG_HTTP_STATUS_PARTIAL_CONTENT,
	multi_status             = NNG_HTTP_STATUS_MULTI_STATUS,
	already_reported         = NNG_HTTP_STATUS_ALREADY_REPORTED,
	im_used                  = NNG_HTTP_STATUS_IM_USED,
	multiple_choices         = NNG_HTTP_STATUS_MULTIPLE_CHOICES,
	moved_permanently        = NNG_HTTP_STATUS_STATUS_MOVED_PERMANENTLY,
	found                    = NNG_HTTP_STATUS_FOUND,
	see_other                = NNG_HTTP_STATUS_SEE_OTHER,
	not_modified             = NNG_HTTP_STATUS_NOT_MODIFIED,
	use_proxy                = NNG_HTTP_STATUS_USE_PROXY,
	temporary_redirect       = NNG_HTTP_STATUS_TEMPORARY_REDIRECT,
	permanent_redirect       = NNG_HTTP_STATUS_PERMANENT_REDIRECT,
	bad_request              = NNG_HTTP_STATUS_BAD_REQUEST,
	unauthorized             = NNG_HTTP_STATUS_UNAUTHORIZED,
	payment_required         = NNG_HTTP_STATUS_PAYMENT_REQUIRED,
	forbidden                = NNG_HTTP_STATUS_FORBIDDEN,
	not_found                = NNG_HTTP_STATUS_NOT_FOUND,
	method_not_allowed       = NNG_HTTP_STATUS_METHOD_NOT_ALLOWED,
	not_acceptable           = NNG_HTTP_STATUS_NOT_ACCEPTABLE,
	proxy_auth_required      = NNG_HTTP_STATUS_PROXY_AUTH_REQUIRED,
	request_timeout          = NNG_HTTP_STATUS_REQUEST_TIMEOUT,
	conflict                 = NNG_HTTP_STATUS_CONFLICT,
	gone                     = NNG_HTTP_STATUS_GONE,
	length_required          = NNG_HTTP_STATUS_LENGTH_REQUIRED,
	precondition_failed      = NNG_HTTP_STATUS_PRECONDITION_FAILED,
	payload_too_large        = NNG_HTTP_STATUS_PAYLOAD_TOO_LARGE,
	entity_too_long          = NNG_HTTP_STATUS_ENTITY_TOO_LONG,
	unsupported_media_type   = NNG_HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE,
	range_not_satisfiable    = NNG_HTTP_STATUS_RANGE_NOT_SATISFIABLE,
	expectation_failed       = NNG_HTTP_STATUS_EXPECTATION_FAILED,
	teapot                   = NNG_HTTP_STATUS_TEAPOT,
	unprocessable_entity     = NNG_HTTP_STATUS_UNPROCESSABLE_ENTITY,
	locked                   = NNG_HTTP_STATUS_LOCKED,
	failed_dependency        = NNG_HTTP_STATUS_FAILED_DEPENDENCY,
	upgrade_required         = NNG_HTTP_STATUS_UPGRADE_REQUIRED,
	precondition_required    = NNG_HTTP_STATUS_PRECONDITION_REQUIRED,
	too_many_requests        = NNG_HTTP_STATUS_TOO_MANY_REQUESTS,
	headers_too_large        = NNG_HTTP_STATUS_HEADERS_TOO_LARGE,
	unavail_legal_reasons    = NNG_HTTP_STATUS_UNAVAIL_LEGAL_REASONS,
	internal_server_error    = NNG_HTTP_STATUS_INTERNAL_SERVER_ERROR,
	not_implemented          = NNG_HTTP_STATUS_NOT_IMPLEMENTED,
	bad_gateway              = NNG_HTTP_STATUS_BAD_GATEWAY,
	service_unavailable      = NNG_HTTP_STATUS_SERVICE_UNAVAILABLE,
	gateway_timeout          = NNG_HTTP_STATUS_GATEWAY_TIMEOUT,
	http_version_not_supp    = NNG_HTTP_STATUS_HTTP_VERSION_NOT_SUPP,
	variant_also_negotiates  = NNG_HTTP_STATUS_VARIANT_ALSO_NEGOTIATES,
	insufficient_storage     = NNG_HTTP_STATUS_INSUFFICIENT_STORAGE,
	loop_detected            = NNG_HTTP_STATUS_LOOP_DETECTED,
	not_extended             = NNG_HTTP_STATUS_NOT_EXTENDED,
	network_auth_required    = NNG_HTTP_STATUS_NETWORK_AUTH_REQUIRED
};

enum class verb {
	get,
	head,
	post,
	put,
	delete_,
	trace,
	options,
	connect,
	patch
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const verb_name[] {
	"GET",
	"HEAD",
	"POST",
	"PUT",
	"DELETE",
	"TRACE",
	"OPTIONS",
	"CONNECT",
	"PATCH"
};

}

inline const char* to_name( verb v ) noexcept {
	return detail::verb_name[(int)v];
}

}}

#endif
