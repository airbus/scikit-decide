/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSER_PASS_HH
#define SKDECIDE_PDDL_PARSER_PASS_HH

#include "pegtl.hpp"
#include "utils/pegtl_spdlog_tracer.hh"
#include "parser_state.hh"

namespace skdecide {

namespace pddl {

namespace parser {

struct DomainStructureTag {};
struct DomainOperatorsTag {};
struct ProblemTag {};
struct NormalControlTag {};
struct TracerControlTag {};

template <typename TgrammarTag, typename TcontrolTag> struct ParsePass {
  static bool run(tao::pegtl::text_file_input<> &in,
                  tao::pegtl::trace_state *ts, state &s);
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSER_PASS_HH
