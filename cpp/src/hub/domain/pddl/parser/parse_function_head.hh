/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_FUNCTION_HEAD_HH
#define SKDECIDE_PDDL_PARSE_FUNCTION_HEAD_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_terms.hh"
#include "function_expression.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct function_symbol : name {};
template <> struct action<function_symbol> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.expression = std::make_shared<FunctionExpression>();
    try {
      std::static_pointer_cast<FunctionExpression>(s.expression)
          ->set_function(s.domain->get_function(s.name));
    } catch (const std::exception &e) {
      if (s.name == "total-cost" &&
          !s.global_requirements->has_action_costs()) {
        throw pegtl::parse_error(
            "using 'total-cost' without enabling :action-costs requirement nor "
            "explicitly defining it",
            in.current_position());
      }
      if (s.name == "total-time" && !s.global_requirements->has_time() &&
          !s.global_requirements->has_durative_actions()) {
        throw pegtl::parse_error(
            "using 'total-time' without enabling :time nor :durative-actions "
            "requirement nor explicitly defining it",
            in.current_position());
      }
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

struct f_class_name : name {};
template <> struct action<f_class_name> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    try {
      s.cls = s.domain->get_class(s.name);
    } catch (const std::exception &e) {
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

struct c_function_symbol : name {};
template <> struct action<c_function_symbol> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    s.expression = std::make_shared<FunctionExpression>();
    try {
      std::static_pointer_cast<FunctionExpression>(s.expression)
          ->set_function(s.cls->get_function(s.name));
    } catch (const std::exception &e) {
      if (s.name == "total-cost" &&
          !s.global_requirements->has_action_costs()) {
        throw pegtl::parse_error(
            "using 'total-cost' without enabling :action-costs requirement nor "
            "explicitly defining it",
            in.current_position());
      }
      if (s.name == "total-time" && !s.global_requirements->has_time() &&
          !s.global_requirements->has_durative_actions()) {
        throw pegtl::parse_error(
            "using 'total-time' without enabling :time nor :durative-actions "
            "requirement nor explicitly defining it",
            in.current_position());
      }
      throw pegtl::parse_error(e.what(), in.current_position());
    }
  }
};

struct function_constant_term : name {};
template <> struct action<function_constant_term> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    std::unordered_map<std::string, Domain::ObjectPtr>::const_iterator i =
        s.registered_objects.find(StringConverter::tolower(s.name));
    if (i == s.registered_objects.end()) {
      throw pegtl::parse_error(
          "object '" + s.name + "' is unknown in the current parsing context",
          in.current_position());
    } else {
      std::static_pointer_cast<FunctionExpression>(s.expression)
          ->append_term(i->second);
    }
  }
};

struct function_variable_term : name {};
template <> struct action<function_variable_term> {
  template <typename Input> static void apply(const Input &in, state &s) {
    s.name = in.string();
    std::unordered_map<std::string, Predicate::VariablePtr>::const_iterator i =
        s.registered_variables.find('?' + StringConverter::tolower(s.name));
    if (i == s.registered_variables.end()) {
      throw pegtl::parse_error(
          "variable '?" + s.name +
              "' is unknown in the current parsing context",
          in.current_position());
    } else {
      std::static_pointer_cast<FunctionExpression>(s.expression)
          ->append_term(i->second);
    }
  }
};

struct function_head
    : pegtl::sor<
          parameter_symbol<function_symbol, function_constant_term,
                           function_variable_term>,
          pegtl::seq<pegtl::one<'('>, ignored, f_class_name, ignored,
                     pegtl::one<'.'>, ignored,
                     parameter_symbol<c_function_symbol, function_constant_term,
                                      function_variable_term>,
                     ignored, pegtl::one<')'>>,
          function_symbol> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_FUNCTION_HEAD_HH
