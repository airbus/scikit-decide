/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_LOGGING_HH
#define SKDECIDE_LOGGING_HH

#include <string>

namespace skdecide {

namespace logging {
enum level { trace, debug, info, warn, err, critical, off };
}

// Proxy interface to spdlog in order to avoid including its headers
// in the files that use the logger
class Logger {
public:
  static void set_level(const logging::level &level);
  static logging::level get_level();
  static bool check_level(const logging::level &level,
                          const std::string &caller);

  static void trace(const std::string &msg);
  static void debug(const std::string &msg);
  static void info(const std::string &msg);
  static void warn(const std::string &msg);
  static void error(const std::string &msg);
  static void critical(const std::string &msg);

private:
  static std::string to_string(const logging::level &level);
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/logging_impl.hh"
#endif

#endif // SKDECIDE_LOGGING_HH
