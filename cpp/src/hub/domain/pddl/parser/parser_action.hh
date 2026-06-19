/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSER_ACTION_HH
#define SKDECIDE_PDDL_PARSER_ACTION_HH

#include "pegtl.hpp"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

// by default a rule has no action

template <typename Rule> struct action : pegtl::nothing<Rule> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSER_ACTION_HH
