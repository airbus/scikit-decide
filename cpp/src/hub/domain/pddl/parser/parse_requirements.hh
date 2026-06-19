/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_REQUIREMENTS_HH
#define SKDECIDE_PDDL_PARSE_REQUIREMENTS_HH

#include "pegtl.hpp"
#include "parser_state.hh"
#include "parser_action.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct req_equality : keyword<':', 'e', 'q', 'u', 'a', 'l', 'i', 't', 'y'> {};

template <> struct action<req_equality> {
  static void apply0(state &s) {
    s.requirements->set_equality();
    s.global_requirements->set_equality();
  }
};

struct req_strips : keyword<':', 's', 't', 'r', 'i', 'p', 's'> {};

template <> struct action<req_strips> {
  static void apply0(state &s) {
    s.requirements->set_strips();
    s.global_requirements->set_strips();
  }
};

struct req_typing : keyword<':', 't', 'y', 'p', 'i', 'n', 'g'> {};

template <> struct action<req_typing> {
  static void apply0(state &s) {
    s.requirements->set_typing();
    s.global_requirements->set_typing();
  }
};

struct req_negative_preconditions
    : keyword<':', 'n', 'e', 'g', 'a', 't', 'i', 'v', 'e', '-', 'p', 'r', 'e',
              'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};

template <> struct action<req_negative_preconditions> {
  static void apply0(state &s) {
    s.requirements->set_negative_preconditions();
    s.global_requirements->set_negative_preconditions();
  }
};

struct req_disjunctive_preconditions
    : keyword<':', 'd', 'i', 's', 'j', 'u', 'n', 'c', 't', 'i', 'v', 'e', '-',
              'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {
};

template <> struct action<req_disjunctive_preconditions> {
  static void apply0(state &s) {
    s.requirements->set_disjunctive_preconditions();
    s.global_requirements->set_disjunctive_preconditions();
  }
};

struct req_existential_preconditions
    : keyword<':', 'e', 'x', 'i', 's', 't', 'e', 'n', 't', 'i', 'a', 'l', '-',
              'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {
};

template <> struct action<req_existential_preconditions> {
  static void apply0(state &s) {
    s.requirements->set_existential_preconditions();
    s.global_requirements->set_existential_preconditions();
  }
};

struct req_universal_preconditions
    : keyword<':', 'u', 'n', 'i', 'v', 'e', 'r', 's', 'a', 'l', '-', 'p', 'r',
              'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};

template <> struct action<req_universal_preconditions> {
  static void apply0(state &s) {
    s.requirements->set_universal_preconditions();
    s.global_requirements->set_universal_preconditions();
  }
};

struct req_conditional_effects
    : keyword<':', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 'a', 'l', '-',
              'e', 'f', 'f', 'e', 'c', 't', 's'> {};

template <> struct action<req_conditional_effects> {
  static void apply0(state &s) {
    s.requirements->set_conditional_effects();
    s.global_requirements->set_conditional_effects();
  }
};

struct req_fluents : keyword<':', 'f', 'l', 'u', 'e', 'n', 't', 's'> {};

template <> struct action<req_fluents> {
  static void apply0(state &s) {
    s.requirements->set_fluents();
    s.global_requirements->set_fluents();
  }
};

struct req_durative_actions
    : keyword<':', 'd', 'u', 'r', 'a', 't', 'i', 'v', 'e', '-', 'a', 'c', 't',
              'i', 'o', 'n', 's'> {};

template <> struct action<req_durative_actions> {
  static void apply0(state &s) {
    s.requirements->set_durative_actions();
    s.global_requirements->set_durative_actions();
    try {
      s.domain->add_function(std::string("total-time"));
    } catch (...) {
      // discard exception thrown when the 'total-time' function is already
      // declared
    }
  }
};

struct req_time : keyword<':', 't', 'i', 'm', 'e'> {};

template <> struct action<req_time> {
  static void apply0(state &s) {
    s.requirements->set_time();
    s.global_requirements->set_time();
    try {
      s.domain->add_function(std::string("total-time"));
    } catch (...) {
      // discard exception thrown when the 'total-time' function is already
      // declared
    }
  }
};

struct req_action_costs
    : keyword<':', 'a', 'c', 't', 'i', 'o', 'n', '-', 'c', 'o', 's', 't', 's'> {
};

template <> struct action<req_action_costs> {
  static void apply0(state &s) {
    s.requirements->set_action_costs();
    s.global_requirements->set_action_costs();
    try {
      s.domain->add_function(std::string("total-cost"));
    } catch (...) {
      // discard exception thrown when the 'total-cost' function is already
      // declared
    }
  }
};

struct req_object_fluents : keyword<':', 'o', 'b', 'j', 'e', 'c', 't', '-', 'f',
                                    'l', 'u', 'e', 'n', 't', 's'> {};

template <> struct action<req_object_fluents> {
  static void apply0(state &s) {
    s.requirements->set_object_fluents();
    s.global_requirements->set_object_fluents();
  }
};

struct req_numeric_fluents : keyword<':', 'n', 'u', 'm', 'e', 'r', 'i', 'c',
                                     '-', 'f', 'l', 'u', 'e', 'n', 't', 's'> {};

template <> struct action<req_numeric_fluents> {
  static void apply0(state &s) {
    s.requirements->set_numeric_fluents();
    s.global_requirements->set_numeric_fluents();
  }
};

struct req_modules : keyword<':', 'm', 'o', 'd', 'u', 'l', 'e', 's'> {};

template <> struct action<req_modules> {
  static void apply0(state &s) {
    s.requirements->set_modules();
    s.global_requirements->set_modules();
  }
};

struct req_adl : keyword<':', 'a', 'd', 'l'> {};

template <> struct action<req_adl> {
  static void apply0(state &s) {
    s.requirements->set_adl();
    s.global_requirements->set_adl();
  }
};

struct req_quantified_preconditions
    : keyword<':', 'q', 'u', 'a', 'n', 't', 'i', 'f', 'i', 'e', 'd', '-', 'p',
              'r', 'e', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n', 's'> {};

template <> struct action<req_quantified_preconditions> {
  static void apply0(state &s) {
    s.requirements->set_quantified_preconditions();
    s.global_requirements->set_quantified_preconditions();
  }
};

struct req_duration_inequalities
    : keyword<':', 'd', 'u', 'r', 'a', 't', 'i', 'o', 'n', '-', 'i', 'n', 'e',
              'q', 'u', 'a', 'l', 'i', 't', 'i', 'e', 's'> {};

template <> struct action<req_duration_inequalities> {
  static void apply0(state &s) {
    s.requirements->set_duration_inequalities();
    s.global_requirements->set_duration_inequalities();
  }
};

struct req_continuous_effects
    : keyword<':', 'c', 'o', 'n', 't', 'i', 'n', 'u', 'o', 'u', 's', '-', 'e',
              'f', 'f', 'e', 'c', 't', 's'> {};

template <> struct action<req_continuous_effects> {
  static void apply0(state &s) {
    s.requirements->set_continuous_effects();
    s.global_requirements->set_continuous_effects();
  }
};

struct req_derived_predicates
    : keyword<':', 'd', 'e', 'r', 'i', 'v', 'e', 'd', '-', 'p', 'r', 'e', 'd',
              'i', 'c', 'a', 't', 'e', 's'> {};

template <> struct action<req_derived_predicates> {
  static void apply0(state &s) {
    s.requirements->set_derived_predicates();
    s.global_requirements->set_derived_predicates();
  }
};

struct req_timed_initial_literals
    : keyword<':', 't', 'i', 'm', 'e', 'd', '-', 'i', 'n', 'i', 't', 'i', 'a',
              'l', '-', 'l', 'i', 't', 'e', 'r', 'a', 'l', 's'> {};

template <> struct action<req_timed_initial_literals> {
  static void apply0(state &s) {
    s.requirements->set_timed_initial_literals();
    s.global_requirements->set_timed_initial_literals();
  }
};

struct req_preferences
    : keyword<':', 'p', 'r', 'e', 'f', 'e', 'r', 'e', 'n', 'c', 'e', 's'> {};

template <> struct action<req_preferences> {
  static void apply0(state &s) {
    s.requirements->set_preferences();
    s.global_requirements->set_preferences();
  }
};

struct req_constraints
    : keyword<':', 'c', 'o', 'n', 's', 't', 'r', 'a', 'i', 'n', 't', 's'> {};

template <> struct action<req_constraints> {
  static void apply0(state &s) {
    s.requirements->set_constraints();
    s.global_requirements->set_constraints();
  }
};

struct req_probabilistic_effects
    : keyword<':', 'p', 'r', 'o', 'b', 'a', 'b', 'i', 'l', 'i', 's', 't', 'i',
              'c', '-', 'e', 'f', 'f', 'e', 'c', 't', 's'> {};

template <> struct action<req_probabilistic_effects> {
  static void apply0(state &s) {
    s.requirements->set_probabilistic_effects();
    s.global_requirements->set_probabilistic_effects();
  }
};

struct req_rewards : keyword<':', 'r', 'e', 'w', 'a', 'r', 'd', 's'> {};

template <> struct action<req_rewards> {
  static void apply0(state &s) {
    s.requirements->set_rewards();
    s.global_requirements->set_rewards();
    try {
      s.domain->add_function(std::string("reward"));
    } catch (...) {
    }
  }
};

struct req_mdp : keyword<':', 'm', 'd', 'p'> {};

template <> struct action<req_mdp> {
  static void apply0(state &s) {
    s.requirements->set_mdp();
    s.global_requirements->set_mdp();
    try {
      s.domain->add_function(std::string("reward"));
    } catch (...) {
    }
  }
};

struct require_key
    : pegtl::sor<req_equality, req_strips, req_typing,
                 req_negative_preconditions, req_disjunctive_preconditions,
                 req_existential_preconditions, req_universal_preconditions,
                 req_conditional_effects, req_fluents, req_durative_actions,
                 req_action_costs, req_object_fluents, req_numeric_fluents,
                 req_modules, req_adl, req_quantified_preconditions,
                 req_duration_inequalities, req_continuous_effects,
                 req_derived_predicates, req_timed_initial_literals,
                 req_time, // sub-string of the previous one so must be parsed
                           // after
                 req_probabilistic_effects, req_preferences, req_constraints,
                 req_rewards, req_mdp> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_REQUIREMENTS_HH
