/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_NAME_HH
#define SKDECIDE_PDDL_PARSE_NAME_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct name
    : pegtl::seq<
          pegtl::plus<pegtl::identifier_first>,
          pegtl::star<pegtl::sor<pegtl::one<'-'>, pegtl::identifier_other>>> {};

template <> struct action<name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
  }
};

template <char... Cs>
struct keyword
    : pegtl::seq<
          pegtl::istring<Cs...>,
          pegtl::not_at<pegtl::sor<pegtl::one<'-'>, pegtl::identifier_other>>> {
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_NAME_HH
