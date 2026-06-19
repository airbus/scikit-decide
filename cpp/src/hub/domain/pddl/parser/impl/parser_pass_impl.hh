/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSER_PASS_IMPL_HH
#define SKDECIDE_PDDL_PARSER_PASS_IMPL_HH

#include <type_traits>

#include "../parser_pass.hh"
#include "../grammar_passes.hh"

namespace pegtl = tao::pegtl; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

template <typename TgrammarTag, typename TcontrolTag>
bool ParsePass<TgrammarTag, TcontrolTag>::run(pegtl::text_file_input<> &in,
                                              pegtl::trace_state *ts,
                                              state &s) {
  if constexpr (std::is_same_v<TgrammarTag, DomainStructureTag>) {
    if constexpr (std::is_same_v<TcontrolTag, NormalControlTag>) {
      return pegtl::parse<domain_structure_pass::grammar, action>(in, s);
    } else {
      return pegtl::parse<domain_structure_pass::grammar, action,
                          pegtl::tracer>(in, *ts, s);
    }
  } else if constexpr (std::is_same_v<TgrammarTag, DomainOperatorsTag>) {
    if constexpr (std::is_same_v<TcontrolTag, NormalControlTag>) {
      return pegtl::parse<domain_operators_pass::grammar, action>(in, s);
    } else {
      return pegtl::parse<domain_operators_pass::grammar, action,
                          pegtl::tracer>(in, *ts, s);
    }
  } else if constexpr (std::is_same_v<TgrammarTag, ProblemTag>) {
    if constexpr (std::is_same_v<TcontrolTag, NormalControlTag>) {
      return pegtl::parse<problem_pass::grammar, action>(in, s);
    } else {
      return pegtl::parse<problem_pass::grammar, action, pegtl::tracer>(in, *ts,
                                                                        s);
    }
  }
}

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSER_PASS_IMPL_HH
