/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_PROBLEM_HH
#define SKDECIDE_PDDL_PARSE_PROBLEM_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_objects.hh"
#include "parse_initial_state.hh"
#include "parse_precondition.hh"
#include "parse_metric.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

// parse problem name

struct problem_name
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<'p', 'r', 'o', 'b', 'l', 'e', 'm'>, ignored>,
          pegtl::seq<name, ignored, pegtl::one<')'>>> {};

template <> struct action<problem_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    auto p = s.problems.emplace(std::make_pair(
        StringConverter::tolower(s.name), std::make_shared<Problem>(s.name)));

    if (!p.second) {
      throw pegtl::parse_error("problem '" + s.name + "' already declared",
                               in.current_position());
    }

    s.problem = p.first->second;
  }
};

// parse problem domain name

struct problem_domain_name
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'd', 'o', 'm', 'a', 'i', 'n'>, ignored>,
          pegtl::seq<name, ignored, pegtl::one<')'>>> {};

template <> struct action<problem_domain_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    auto d = s.domains.find(StringConverter::tolower(s.name));

    if (d == s.domains.end()) {
      throw pegtl::parse_error("no existing domain '" + s.name +
                                   "' when parsing problem '" +
                                   s.problem->get_name() + "'",
                               in.current_position());
    }

    s.problem->set_domain(d->second);
    s.domain = s.problem->get_domain();
    s.global_requirements =
        (s.domain->get_requirements())
            ? (s.domain->get_requirements())
            : (std::make_shared<Requirements>()); // exists even if no declared
                                                  // requirements
    for (const auto &o : s.domain->get_objects()) {
      s.registered_objects.insert(std::make_pair(o->get_name(), o));
    }
  }
};

// parse problem requirement definition

struct open_problem_requirements
    : pegtl::seq<pegtl::one<'('>, ignored,
                 keyword<':', 'r', 'e', 'q', 'u', 'i', 'r', 'e', 'm', 'e', 'n',
                         't', 's'>,
                 ignored> {};

template <> struct action<open_problem_requirements> {
  static void apply0(state &s) {
    s.requirements = std::make_shared<Requirements>();
  }
};

struct problem_require_def
    : pegtl::if_must<open_problem_requirements,
                     pegtl::seq<pegtl::list<require_key, ignored>, ignored,
                                pegtl::one<')'>>> {};

template <> struct action<problem_require_def> {
  static void apply0(state &s) { s.problem->set_requirements(s.requirements); }
};

// parse problem objects

struct problem_objects
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'o', 'b', 'j', 'e', 'c', 't', 's'>, ignored>,
          pegtl::seq<object_list, ignored, pegtl::one<')'>>> {};

// parse initial state

struct open_initial_state
    : pegtl::seq<pegtl::one<'('>, ignored, keyword<':', 'i', 'n', 'i', 't'>,
                 ignored> {};
template <> struct action<open_initial_state> {
  static void apply0(state &s) {
    s.effects.push(std::make_shared<ConjunctionEffect>());
  }
};

struct close_initial_state : pegtl::seq<init_els, ignored, pegtl::one<')'>> {};

template <> struct action<close_initial_state> {
  static void apply0(state &s) {
    s.problem->set_initial_effect(
        std::static_pointer_cast<ConjunctionEffect>(s.effects.top()));
    s.effects.pop();
  }
};

struct initial_state : pegtl::if_must<open_initial_state, close_initial_state> {
};

// parse goal

struct goal_spec
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<':', 'g', 'o', 'a', 'l'>, ignored>,
                     pegtl::seq<precondition, ignored, pegtl::one<')'>>> {};

template <> struct action<goal_spec> {
  static void apply0(state &s) { s.problem->set_goal(s.formula); }
};

// parse constraints

struct constraints_probdef
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<':', 'c', 'o', 'n', 's', 't', 'r', 'a',
                                        'i', 'n', 't', 's'>,
                                ignored>,
                     pegtl::seq<pref_con_goal, ignored, pegtl::one<')'>>> {};

template <> struct action<constraints_probdef> {
  static void apply0(state &s) { s.problem->set_constraints(s.formula); }
};

// parse optimization metric

struct metric_spec
    : pegtl::if_must<
          pegtl::seq<pegtl::one<'('>, ignored,
                     keyword<':', 'm', 'e', 't', 'r', 'i', 'c'>, ignored>,
          pegtl::seq<metric, ignored, pegtl::one<')'>>> {};

template <> struct action<metric_spec> {
  static void apply0(state &s) { s.problem->set_metric(s.expression); }
};

// parse goal-reward

struct goal_reward_spec
    : pegtl::if_must<pegtl::seq<pegtl::one<'('>, ignored,
                                keyword<':', 'g', 'o', 'a', 'l', '-', 'r', 'e',
                                        'w', 'a', 'r', 'd'>,
                                ignored>,
                     pegtl::seq<ground_f_exp, ignored, pegtl::one<')'>>> {};

template <> struct action<goal_reward_spec> {
  static void apply0(state &s) { s.problem->set_goal_reward(s.expression); }
};

// parse problem body

struct problem_body_item
    : pegtl::sor<problem_require_def, problem_objects, initial_state, goal_spec,
                 constraints_probdef, goal_reward_spec, metric_spec> {};

struct problem_body : pegtl::list<problem_body_item, ignored> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_PROBLEM_HH
