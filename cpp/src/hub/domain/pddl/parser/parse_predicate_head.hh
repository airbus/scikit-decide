/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_PROPOSITION_HH
#define SKDECIDE_PDDL_PARSE_PROPOSITION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_name.hh"
#include "parse_terms.hh"

#include "predicate_formula.hh"
#include "predicate_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct formula;
struct effect;

template <typename PredicateRule, typename Enable = void> struct PredicateProxy;

template <typename PredicateRule>
struct PredicateProxy<PredicateRule,
                      typename std::enable_if<
                          std::is_same<PredicateRule, formula>::value>::type> {
  typedef PredicateFormula PDDLType;
  static Formula::Ptr &last_parsed(state &s) { return s.formula; }
};

template <typename PredicateRule>
struct PredicateProxy<
    PredicateRule,
    typename std::enable_if<std::is_same<PredicateRule, effect>::value>::type> {
  typedef PredicateEffect PDDLType;
  static Effect::Ptr &last_parsed(state &s) { return s.effect; }
};

template <typename PredicateRule>
struct predicate_symbol : pegtl::action<action, name> {
  typedef PredicateProxy<PredicateRule> PP;
};

template <typename Rule> struct predicate_symbol_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::PP PP;
    s.name = in.string();
    PP::last_parsed(s) = std::make_shared<typename PP::PDDLType>();
    try {
      std::static_pointer_cast<typename PP::PDDLType>(PP::last_parsed(s))
          ->set_predicate(s.domain->get_predicate(s.name));
    } catch (...) {
      try {
        std::static_pointer_cast<typename PP::PDDLType>(PP::last_parsed(s))
            ->set_predicate(s.domain->get_derived_predicate(s.name));
      } catch (const std::exception &e) {
        throw pegtl::parse_error(
            "no predicate neither derived predicate with name '" + s.name +
                "' found in domain '" + s.domain->get_name() + "'",
            in.current_position());
      }
    }
  }
};

template <typename PredicateRule>
struct predicate_constant_term : pegtl::action<action, name> {
  typedef PredicateProxy<PredicateRule> PP;
};

template <typename Rule> struct predicate_constant_term_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::PP PP;
    s.name = in.string();
    std::unordered_map<std::string, Domain::ObjectPtr>::const_iterator i =
        s.registered_objects.find(StringConverter::tolower(s.name));
    if (i == s.registered_objects.end()) {
      throw pegtl::parse_error("object '" + s.name +
                                   "' unknown to the current parsing context",
                               in.current_position());
    } else {
      std::static_pointer_cast<typename PP::PDDLType>(PP::last_parsed(s))
          ->append_term(i->second);
    }
  }
};

template <typename PredicateRule>
struct predicate_variable_term : pegtl::action<action, name> {
  typedef PredicateProxy<PredicateRule> PP;
};

template <typename Rule> struct predicate_variable_term_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::PP PP;
    s.name = in.string();
    std::unordered_map<std::string, Predicate::VariablePtr>::const_iterator i =
        s.registered_variables.find('?' + StringConverter::tolower(s.name));
    if (i == s.registered_variables.end()) {
      throw pegtl::parse_error("variable '?" + s.name +
                                   "' unknown to the current parsing context",
                               in.current_position());
    } else {
      std::static_pointer_cast<typename PP::PDDLType>(PP::last_parsed(s))
          ->append_term(i->second);
    }
  }
};

template <typename PredicateRule>
struct predicate_head
    : parameter_symbol<pegtl::action<predicate_symbol_action,
                                     predicate_symbol<PredicateRule>>,
                       pegtl::action<predicate_constant_term_action,
                                     predicate_constant_term<PredicateRule>>,
                       pegtl::action<predicate_variable_term_action,
                                     predicate_variable_term<PredicateRule>>> {
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_PROPOSITION_HH
