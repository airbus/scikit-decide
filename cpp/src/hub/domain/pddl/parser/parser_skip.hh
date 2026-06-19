/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSER_SKIP_HH
#define SKDECIDE_PDDL_PARSER_SKIP_HH

#include "pegtl.hpp"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

// Ignore spaces and comments

struct comment : pegtl::if_must<pegtl::one<';'>, pegtl::until<pegtl::eolf>> {};

struct ignored : pegtl::star<pegtl::sor<pegtl::space, comment>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSER_SKIP_HH
