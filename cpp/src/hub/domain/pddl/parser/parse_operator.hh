/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_ACTION_HH
#define SKDECIDE_PDDL_PARSE_ACTION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"
#include "parse_terms.hh"
#include "parse_precondition.hh"
#include "parse_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct ActionOperator {
  typedef keyword<':', 'a', 'c', 't', 'i', 'o', 'n'> kw;
  static Action::Ptr create_operator(state &s) {
    return std::make_shared<Action>(s.name);
  }
  static void register_operator(state &s) {
    s.domain->add_action(get_operator(s));
  }
  static Action::Ptr &get_operator(state &s) { return s.action; }
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {}
};

struct DurativeActionOperator {
  typedef keyword<':', 'd', 'u', 'r', 'a', 't', 'i', 'v', 'e', '-', 'a', 'c',
                  't', 'i', 'o', 'n'>
      kw;
  static DurativeAction::Ptr create_operator(state &s) {
    return std::make_shared<DurativeAction>(s.name);
  }
  static void register_operator(state &s) {
    s.domain->add_durative_action(get_operator(s));
  }
  static DurativeAction::Ptr &get_operator(state &s) {
    return s.durative_action;
  }
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_durative_actions()) {
      throw pegtl::parse_error("defining durative action without enabling "
                               ":durative-actions requirement",
                               in.current_position());
    }
  }
};

struct EventOperator {
  typedef keyword<':', 'e', 'v', 'e', 'n', 't'> kw;
  static Event::Ptr create_operator(state &s) {
    return std::make_shared<Event>(s.name);
  }
  static void register_operator(state &s) {
    s.domain->add_event(get_operator(s));
  }
  static Event::Ptr &get_operator(state &s) { return s.event; }
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_time()) {
      throw pegtl::parse_error(
          "defining event without enabling :time requirement",
          in.current_position());
    }
  }
};

struct ProcessOperator {
  typedef keyword<':', 'p', 'r', 'o', 'c', 'e', 's', 's'> kw;
  static Process::Ptr create_operator(state &s) {
    return std::make_shared<Process>(s.name);
  }
  static void register_operator(state &s) {
    s.domain->add_process(get_operator(s));
  }
  static Process::Ptr &get_operator(state &s) { return s.process; }
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_time()) {
      throw pegtl::parse_error(
          "defining process without enabling :time requirement",
          in.current_position());
    }
  }
};

template <typename Operator>
struct operator_name : pegtl::action<action, name> {
  typedef Operator OP;
};

template <typename Rule> struct operator_name_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::OP OP;
    s.name = in.string();
    OP::get_operator(s) = OP::create_operator(s);
  }
};

template <typename Operator>
struct open_operator
    : pegtl::action<action, pegtl::seq<pegtl::one<'('>, ignored,
                                       typename Operator::kw, ignored>> {
  typedef Operator OP;
};

template <typename Rule> struct open_operator_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::OP OP;
    OP::check_requirement(in, s);
  }
};

struct open_operator_parameters
    : pegtl::seq<keyword<':', 'p', 'a', 'r', 'a', 'm', 'e', 't', 'e', 'r', 's'>,
                 ignored, pegtl::one<'('>, ignored> {};

template <typename Operator>
struct parameter_variable_operator : pegtl::action<action, name> {
  typedef Operator OP;
};

template <typename Rule> struct parameter_variable_operator_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::OP OP;
    s.name = in.string();
    s.variable_list.push_back(OP::get_operator(s)->append_variable(s.name));
    auto i = s.registered_variables.insert(std::make_pair(
        '?' + StringConverter::tolower(s.name), s.variable_list.back()));
    if (!i.second) {
      throw pegtl::parse_error(
          "action variable '?" + s.name +
              "' already existing in the current parsing context",
          in.current_position());
    }
  }
};

template <typename Operator>
struct close_operator_parameters
    : pegtl::seq<
          typed_var_list<pegtl::action<parameter_variable_operator_action,
                                       parameter_variable_operator<Operator>>>,
          ignored, pegtl::one<')'>> {};

template <typename Operator>
struct open_operator_precondition
    : pegtl::seq<typename std::conditional<
                     std::is_same<Operator, DurativeActionOperator>::value,
                     keyword<':', 'c', 'o', 'n', 'd', 'i', 't', 'i', 'o', 'n'>,
                     keyword<':', 'p', 'r', 'e', 'c', 'o', 'n', 'd', 'i', 't',
                             'i', 'o', 'n'>>::type,
                 ignored> {};

template <typename Operator>
struct close_operator_precondition
    : pegtl::action<
          action,
          typename std::conditional<
              std::is_same<Operator, ActionOperator>::value, precondition,
              typename std::conditional<
                  std::is_same<Operator, DurativeActionOperator>::value,
                  timed_precondition, formula>::type>::type> {
  typedef Operator OP;
};

template <typename Rule> struct close_operator_precondition_action {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    OP::get_operator(s)->set_condition(s.formula);
  }
};

struct open_operator_effect
    : pegtl::seq<keyword<':', 'e', 'f', 'f', 'e', 'c', 't'>, ignored> {};

template <typename Operator>
struct close_operator_effect
    : pegtl::action<
          action,
          typename std::conditional<
              std::is_same<Operator, ProcessOperator>::value, process_effect,
              typename std::conditional<
                  std::is_same<Operator, DurativeActionOperator>::value,
                  da_effect, effect>::type>::type> {
  typedef Operator OP;
};

template <typename Rule> struct close_operator_effect_action {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    OP::get_operator(s)->set_effect(s.effect);
  }
};

template <typename Operator>
struct close_operator
    : pegtl::action<
          action,
          pegtl::seq<
              pegtl::action<operator_name_action, operator_name<Operator>>,
              ignored,
              pegtl::opt<pegtl::if_must<open_operator_parameters,
                                        close_operator_parameters<Operator>>>,
              typename std::conditional<
                  std::is_same<Operator, DurativeActionOperator>::value,
                  pegtl::seq<
                      ignored,
                      keyword<':', 'd', 'u', 'r', 'a', 't', 'i', 'o', 'n'>,
                      ignored, duration_constraint, ignored>,
                  ignored>::type,
              pegtl::opt<pegtl::if_must<
                  open_operator_precondition<Operator>,
                  pegtl::action<close_operator_precondition_action,
                                close_operator_precondition<Operator>>>>,
              ignored,
              pegtl::opt<pegtl::if_must<
                  open_operator_effect,
                  pegtl::action<close_operator_effect_action,
                                close_operator_effect<Operator>>>>,
              ignored, pegtl::one<')'>>> {
  typedef Operator OP;
};

template <typename Rule> struct close_operator_action {
  static void apply0(state &s) {
    typedef typename Rule::OP OP;
    if (!OP::get_operator(s)->get_condition()) {
      OP::get_operator(s)->set_condition(
          std::make_shared<ConjunctionFormula>());
    }
    if (!OP::get_operator(s)->get_effect()) {
      OP::get_operator(s)->set_effect(std::make_shared<ConjunctionEffect>());
    }
    OP::register_operator(s);
    for (const auto &v : OP::get_operator(s)->get_variables()) {
      s.registered_variables.erase(v->get_name());
    }
  }
};

template <typename Operator>
struct operator_def
    : pegtl::if_must<
          pegtl::action<open_operator_action, open_operator<Operator>>,
          pegtl::action<close_operator_action, close_operator<Operator>>> {};
} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_ACTION_HH
