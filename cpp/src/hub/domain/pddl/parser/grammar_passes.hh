/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_GRAMMAR_PASSES_HH
#define SKDECIDE_PDDL_GRAMMAR_PASSES_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"
#include "parser_skip.hh"
#include "parse_name.hh"
#include "parse_domain.hh"
#include "parse_problem.hh"
#include "utils/pegtl_spdlog_tracer.hh"

#include "spdlog/spdlog.h"

namespace pegtl = tao::pegtl; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

// Balanced paren skip: matches any balanced parenthesized expression
// Used to skip blocks the current pass doesn't process
struct balanced_skip;
struct balanced_content : pegtl::sor<balanced_skip, pegtl::not_one<'(', ')'>> {
};
struct balanced_skip
    : pegtl::seq<pegtl::one<'('>, pegtl::star<balanced_content>,
                 pegtl::one<')'>> {};

// Like pegtl::try_catch_return_false but logs caught exceptions as warnings
template <typename Rule> struct try_catch_warn_return_false {
  using rule_t = try_catch_warn_return_false;
  using subs_t = pegtl::type_list<Rule>;

  template <pegtl::apply_mode A, pegtl::rewind_mode M,
            template <typename...> class Action,
            template <typename...> class Control, typename ParseInput,
            typename... States>
  [[nodiscard]] static bool match(ParseInput &in, States &&...st) {
    auto m = Control<try_catch_warn_return_false>::template guard<A, M, Action,
                                                                  Control>(
        in, st...);
    try {
      return m(Control<Rule>::template match<A, pegtl::rewind_mode::optional,
                                             Action, Control>(in, st...));
    } catch (const pegtl::parse_error_base &e) {
      spdlog::warn("PDDL parse error (recovered): {}", e.what());
      return false;
    }
  }
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

// Disable PEGTL control for try_catch_warn_return_false (matches PEGTL's own
// try_catch_return_false behavior)
namespace TAO_PEGTL_NAMESPACE::internal {
template <typename Rule>
inline constexpr bool
    enable_control<skdecide::pddl::parser::try_catch_warn_return_false<Rule>> =
        false;
} // namespace TAO_PEGTL_NAMESPACE::internal

namespace skdecide {

namespace pddl {

namespace parser {

// ========================================================================
// Pass 1: Domain Structure (types, predicates, functions, constants, etc.)
// Skips operator definitions using balanced_skip.
// ========================================================================

namespace domain_structure_pass {

struct preamble_item
    : pegtl::sor<domain_require_def, type_names, domain_constants, predicates,
                 functions_def, constraints_def, classes, balanced_skip> {};

struct preamble : pegtl::list<preamble_item, ignored> {};

struct domain : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                          keyword<'d', 'e', 'f', 'i', 'n', 'e'>,
                                          ignored, domain_name, ignored>,
                               pegtl::seq<preamble, ignored, pegtl::one<')'>>> {
};

struct try_domain : try_catch_warn_return_false<domain> {};

struct grammar
    : pegtl::star<
          pegtl::seq<ignored, pegtl::sor<try_domain, balanced_skip>, ignored>> {
};

} // namespace domain_structure_pass

template <> struct action<domain_structure_pass::domain> {
  static void apply0(state &s) {
    s.domain.reset();
    s.registered_objects.clear();
  }
};

// ========================================================================
// Pass 2: Domain Operators (actions, events, processes, derivations)
// Skips structure declarations using balanced_skip.
// Uses a lookup action for domain_name (domain created in pass 1).
// ========================================================================

namespace domain_operators_pass {

struct domain_name_lookup : domain_name {};

struct preamble_item : pegtl::sor<structure_defs, balanced_skip> {};

struct preamble : pegtl::list<preamble_item, ignored> {};

struct domain : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                          keyword<'d', 'e', 'f', 'i', 'n', 'e'>,
                                          ignored, domain_name_lookup, ignored>,
                               pegtl::seq<preamble, ignored, pegtl::one<')'>>> {
};

struct try_domain : try_catch_warn_return_false<domain> {};

struct grammar
    : pegtl::star<
          pegtl::seq<ignored, pegtl::sor<try_domain, balanced_skip>, ignored>> {
};

} // namespace domain_operators_pass

template <> struct action<domain_operators_pass::domain_name_lookup> {
  template <typename Input> static void apply(const Input &in, state &s) {
    auto d = s.domains.find(StringConverter::tolower(s.name));
    if (d == s.domains.end()) {
      throw pegtl::parse_error("domain '" + s.name + "' not found",
                               in.current_position());
    }
    s.domain = d->second;
    s.global_requirements = s.domain->get_requirements()
                                ? s.domain->get_requirements()
                                : std::make_shared<Requirements>();
    for (const auto &o : s.domain->get_objects()) {
      s.registered_objects.insert(std::make_pair(o->get_name(), o));
    }
  }
};

template <> struct action<domain_operators_pass::domain> {
  static void apply0(state &s) {
    s.domain.reset();
    s.registered_objects.clear();
  }
};

// ========================================================================
// Pass 3: Problem
// Skips domain blocks using balanced_skip.
// ========================================================================

namespace problem_pass {

struct problem
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<'d', 'e', 'f', 'i', 'n', 'e'>, ignored,
                     problem_name, ignored, problem_domain_name, ignored>,
          pegtl::seq<problem_body, ignored, pegtl::one<')'>>> {};

struct try_problem : try_catch_warn_return_false<problem> {};

struct grammar
    : pegtl::star<pegtl::seq<ignored, pegtl::sor<try_problem, balanced_skip>,
                             ignored>> {};

} // namespace problem_pass

template <> struct action<problem_pass::problem> {
  static void apply0(state &s) {
    s.problem.reset();
    s.registered_objects.clear();
  }
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_GRAMMAR_PASSES_HH
