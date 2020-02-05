#ifndef NNGPP_TRANSPORT_IPC_H
#define NNGPP_TRANSPORT_IPC_H
#include <nngpp/listener_view.h>
#include <nngpp/pipe_view.h>
#include <nng/transport/ipc/ipc.h>

namespace nng { namespace ipc {

inline void register_transport() {
	int r = nng_ipc_register();
	if( r != 0 ) {
		throw exception(r,"nng_ipc_register");
	}
}

enum class option {
	peer_uid,            // u64
	peer_gid,            // u64
	peer_pid,            // u64
	peer_zoneid,         // u64
	security_descriptor, // pointer
	permissions          // int
};

namespace detail {

#if __cpp_inline_variables >= 201606
inline
#else
static
#endif
const char* const option_names[] = {
	NNG_OPT_IPC_PEER_UID,
	NNG_OPT_IPC_PEER_GID,
	NNG_OPT_IPC_PEER_PID,
	NNG_OPT_IPC_PEER_ZONEID,
	NNG_OPT_IPC_SECURITY_DESCRIPTOR,
	NNG_OPT_IPC_PERMISSIONS
};

}

inline const char* to_name( option o ) noexcept {
	return detail::option_names[(int)o];
}

inline uint64_t get_opt_peer_uid( nng::pipe_view s ) {
	return s.get_opt_uint64( to_name(option::peer_uid) );
}

inline uint64_t get_opt_peer_gid( nng::pipe_view s ) {
	return s.get_opt_uint64( to_name(option::peer_gid) );
}

inline uint64_t get_opt_peer_pid( nng::pipe_view s ) {
	return s.get_opt_uint64( to_name(option::peer_pid) );
}

inline uint64_t get_opt_peer_zoneid( nng::pipe_view s ) {
	return s.get_opt_uint64( to_name(option::peer_zoneid) );
}

inline void set_opt_security_descriptor( nng::listener_view s, void* v ) {
	s.set_opt_ptr( to_name(option::security_descriptor), v );
}

inline void set_opt_permissions( nng::listener_view s, int v ) {
	s.set_opt_int( to_name(option::permissions), v );
}

}}

#endif
