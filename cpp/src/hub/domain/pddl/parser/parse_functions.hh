/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_FUNCTIONS_HH
#define SKDECIDE_PDDL_PARSE_FUNCTIONS_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_terms.hh"

#include "function.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct domain_function_name : name {};
template <> struct action<domain_function_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    if ((!s.global_requirements->has_action_costs() ||
         (s.name != "total-cost")) &&
        (!s.global_requirements->has_time() || (s.name != "total-time")) &&
        (!s.global_requirements->has_rewards() || (s.name != "reward"))) {
      s.function = s.domain->add_function(s.name);
    }
  }
};

struct class_function_name : name {};
template <> struct action<class_function_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.function = s.cls->add_function(s.name);
  }
};

struct function_variable : name {};
template <> struct action<function_variable> {
  template <typename Input> static void apply(const Input &in, state &s) {
    if (s.function->get_name() == "total-cost" ||
        s.function->get_name() == "total-time" ||
        s.function->get_name() == "reward") {
      throw pegtl::parse_error("reserved functions 'total-cost', "
                               "'total-time' and 'reward' do not have "
                               "parameters",
                               in.current_position());
    }
    s.name = in.string();
    s.variable_list.push_back(s.function->append_variable(s.name));
  }
};

struct function_type_spec
    : pegtl::if_must<pegtl::seq<pegtl::one<'-'>, ignored>,
                     keyword<'n', 'u', 'm', 'b', 'e', 'r'>> {};

template <typename FunctionName>
struct function_typed_list
    : pegtl::seq<
          pegtl::list<term_symbol<FunctionName, function_variable>, ignored>,
          ignored, pegtl::opt<function_type_spec>> {};

template <typename Rule> struct function_list_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    // check if the domain has other functions than total-time and total-cost
    // without delcaring numeric fluents requirement, or if it has the two later
    // functions but without declaring
    //  action costs or time requirements, in which case numeric fluents are
    //  required
    for (const auto &f : s.domain->get_functions()) {
      if (((f->get_name() != "total-cost") ||
           ((f->get_name() == "total-cost") &&
            !s.global_requirements->has_action_costs())) &&
          ((f->get_name() != "total-time") ||
           ((f->get_name() == "total-time") &&
            !s.global_requirements->has_time() &&
            !s.global_requirements->has_durative_actions())) &&
          ((f->get_name() != "reward") ||
           ((f->get_name() == "reward") &&
            !s.global_requirements->has_rewards())) &&
          !s.global_requirements->has_numeric_fluents()) {
        throw pegtl::parse_error(
            "declaring functions without :numeric-fluents requirement",
            in.current_position());
      }
    }
  }
};

template <typename FunctionName>
struct function_list
    : pegtl::action<
          action,
          pegtl::star<pegtl::seq<function_typed_list<FunctionName>, ignored>>> {
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_FUNCTIONS_HH
