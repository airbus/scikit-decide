/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_LOGGING_IMPL_HH
#define SKDECIDE_LOGGING_IMPL_HH

#include <exception>
#include <iostream>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace skdecide {

void Logger::set_level(const logging::level &level) {
  switch (level) {
  case logging::trace:
    spdlog::set_level(spdlog::level::trace);
    break;

  case logging::debug:
    spdlog::set_level(spdlog::level::debug);
    break;

  case logging::info:
    spdlog::set_level(spdlog::level::info);
    break;

  case logging::warn:
    spdlog::set_level(spdlog::level::warn);
    break;

  case logging::err:
    spdlog::set_level(spdlog::level::err);
    break;

  case logging::critical:
    spdlog::set_level(spdlog::level::critical);
    break;

  case logging::off:
    spdlog::set_level(spdlog::level::off);
    break;
  }
}

logging::level Logger::get_level() {
  switch (spdlog::get_level()) {
  case spdlog::level::trace:
    return logging::trace;

  case spdlog::level::debug:
    return logging::debug;

  case spdlog::level::info:
    return logging::info;

  case spdlog::level::warn:
    return logging::warn;

  case spdlog::level::err:
    return logging::err;

  case spdlog::level::critical:
    return logging::critical;

  case spdlog::level::off:
    return logging::off;

  default:
    throw std::runtime_error("Unknown spdlog loggin level " +
                             std::to_string(spdlog::get_level()));
  }
}

bool Logger::check_level(const logging::level &level,
                         const std::string &caller) {
  logging::level actual_level = get_level();
  if (actual_level > level) {
    std::string level_str = to_string(level);
    std::string msg = level_str + " logs requested for " + caller +
                      " but global log level [==" + to_string(actual_level) +
                      "] is higher than " + level_str;
    if (actual_level <= logging::warn) {
      spdlog::warn(msg);
    } else {
      msg = "\033[1;33mbold " + msg + "\033[0m";
      std::cerr << msg << std::endl;
    }
    return false;
  } else {
    return true;
  }
}

std::string Logger::to_string(const logging::level &level) {
  switch (level) {
  case logging::trace:
    return "trace";

  case logging::debug:
    return "debug";

  case logging::info:
    return "info";

  case logging::warn:
    return "warning";

  case logging::err:
    return "error";

  case logging::critical:
    return "critical";

  case logging::off:
    return "off";

  default:
    return "";
  }
}

void Logger::trace(const std::string &msg) { spdlog::trace(msg); }

void Logger::debug(const std::string &msg) { spdlog::debug(msg); }

void Logger::info(const std::string &msg) { spdlog::info(msg); }

void Logger::warn(const std::string &msg) { spdlog::warn(msg); }

void Logger::error(const std::string &msg) { spdlog::error(msg); }

void Logger::critical(const std::string &msg) { spdlog::critical(msg); }

} // namespace skdecide

#endif // SKDECIDE_LOGGING_IMPL_HH
