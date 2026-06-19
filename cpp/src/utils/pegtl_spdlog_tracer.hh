/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) 2014-2026 Dr. Colin Hirsch and Daniel Frey
// Please see LICENSE for license or visit https://github.com/taocpp/PEGTL/

#ifndef _SKDECIDE_PEGTL_SPDLOG_TRACER_HH
#define _SKDECIDE_PEGTL_SPDLOG_TRACER_HH

#include <cassert>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include "pegtl/config.hpp"
#include "pegtl/normal.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace tao {
namespace pegtl {
template <typename Input>
void print_current(const Input &in, std::ostringstream &o) {
  if (in.empty()) {
    o << "<eof>";
  } else {
    const auto c = in.peek_uint8();
    switch (c) {
    case 0:
      o << "<nul> = ";
      break;
    case 9:
      o << "<ht> = ";
      break;
    case 10:
      o << "<lf> = ";
      break;
    case 13:
      o << "<cr> = ";
      break;
    default:
      if (isprint(c)) {
        o << '\'' << c << "' = ";
      }
    }
    o << "(char)" << unsigned(c);
  }
}

struct trace_state {
  unsigned rule = 0;
  unsigned line = 0;
  std::vector<unsigned> stack;
};

template <template <typename...> class Base> struct trace {
  template <typename Rule> struct control : Base<Rule> {
    template <typename Input, typename... States>
    static void start(const Input &in, trace_state &ts, States &&...st) {
      std::ostringstream o;
      o << std::setw(6) << ++ts.line << " " << std::setw(6) << ++ts.rule << " ";
      o << in.current_position() << "  start  " << demangle<Rule>()
        << "; current ";
      print_current(in, o);
      spdlog::debug(o.str());
      Base<Rule>::start(in, st...);
      ts.stack.push_back(ts.rule);
    }

    template <typename Input, typename... States>
    static void success(const Input &in, trace_state &ts, States &&...st) {
      assert(!ts.stack.empty());
      std::ostringstream o;
      o << std::setw(6) << ++ts.line << " " << std::setw(6) << ts.stack.back()
        << " ";
      o << in.current_position() << " success " << demangle<Rule>()
        << "; next ";
      print_current(in, o);
      spdlog::debug(o.str());
      Base<Rule>::success(in, st...);
      ts.stack.pop_back();
    }

    template <typename Input, typename... States>
    static void failure(const Input &in, trace_state &ts, States &&...st) {
      assert(!ts.stack.empty());
      std::ostringstream o;
      o << std::setw(6) << ++ts.line << " " << std::setw(6) << ts.stack.back()
        << " ";
      o << in.current_position() << " failure " << demangle<Rule>();
      spdlog::debug(o.str());
      Base<Rule>::failure(in, st...);
      ts.stack.pop_back();
    }

    template <template <typename...> class Action, typename RewindPosition,
              typename Input, typename... States>
    static auto apply(const RewindPosition &begin, const Input &in,
                      trace_state &ts, States &&...st)
        -> decltype(Base<Rule>::template apply<Action>(begin, in, st...)) {
      std::ostringstream o;
      o << std::setw(6) << ++ts.line << "        ";
      o << in.current_position() << "  apply  " << demangle<Rule>();
      spdlog::debug(o.str());
      return Base<Rule>::template apply<Action>(begin, in, st...);
    }

    template <template <typename...> class Action, typename Input,
              typename... States>
    static auto apply0(const Input &in, trace_state &ts, States &&...st)
        -> decltype(Base<Rule>::template apply0<Action>(in, st...)) {
      std::ostringstream o;
      o << std::setw(6) << ++ts.line << "        ";
      o << in.current_position() << "  apply0 " << demangle<Rule>();
      spdlog::debug(o.str());
      return Base<Rule>::template apply0<Action>(in, st...);
    }
  };
};

template <typename Rule> using tracer = trace<normal>::control<Rule>;

} // namespace pegtl

} // namespace tao

#endif // SKDECIDE_PEGTL_SPDLOG_TRACER_HH
