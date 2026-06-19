/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_QUANTIFICATION_HH
#define SKDECIDE_PDDL_PARSE_QUANTIFICATION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_terms.hh"

#include "quantified_formula.hh"
#include "quantified_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct constraint_goal;
struct pref_goal;
struct pref_con_goal;
struct formula;
struct timed_precondition;
struct effect;
struct da_effect;
template <typename FormulaRule> struct precondition_preference;

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule, typename Enable = void>
struct QuantificationProxy;

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule>
struct QuantificationProxy<
    QuantificationOperator, QuantifiedRule,
    typename std::enable_if<
        std::is_same<QuantifiedRule, constraint_goal>::value ||
        std::is_same<QuantifiedRule, pref_goal>::value ||
        std::is_same<QuantifiedRule, pref_con_goal>::value ||
        std::is_same<QuantifiedRule, formula>::value ||
        std::is_same<QuantifiedRule, timed_precondition>::value ||
        std::is_same<QuantifiedRule,
                     precondition_preference<formula>>::value>::type> {
  typedef Formula BasePDDLType;
  typedef typename QuantificationOperator<BasePDDLType>::type DerivedPDDLType;
  static std::stack<Formula::Ptr> &parsing_stack(state &s) {
    return s.formulas;
  }
  static Formula::Ptr &last_parsed(state &s) { return s.formula; }
  static void set_quantified(typename DerivedPDDLType::Ptr u, Formula::Ptr f) {
    u->set_formula(f);
  }
};

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule>
struct QuantificationProxy<
    QuantificationOperator, QuantifiedRule,
    typename std::enable_if<
        std::is_same<QuantifiedRule, effect>::value ||
        std::is_same<QuantifiedRule, da_effect>::value>::type> {
  typedef Effect BasePDDLType;
  typedef typename QuantificationOperator<BasePDDLType>::type DerivedPDDLType;
  static std::stack<Effect::Ptr> &parsing_stack(state &s) { return s.effects; }
  static Effect::Ptr &last_parsed(state &s) { return s.effect; }
  static void set_quantified(typename DerivedPDDLType::Ptr u, Effect::Ptr e) {
    u->set_effect(e);
  }
};

template <typename BasePDDLType> struct Universal;

template <> struct Universal<Formula> {
  typedef keyword<'f', 'o', 'r', 'a', 'l', 'l'> kw;
  typedef UniversalFormula type;

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_universal_preconditions()) {
      throw pegtl::parse_error("using 'forall' formula without enabling "
                               ":universal-preconditions requirement",
                               in.current_position());
    }
  }
};

template <> struct Universal<Effect> {
  typedef keyword<'f', 'o', 'r', 'a', 'l', 'l'> kw;
  typedef UniversalEffect type;

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_conditional_effects()) {
      throw pegtl::parse_error("using 'forall' effect without enabling "
                               ":conditional-effects requirement",
                               in.current_position());
    }
  }
};

template <typename BasePDDLType> struct Existential;

template <> struct Existential<Formula> {
  typedef keyword<'e', 'x', 'i', 's', 't', 's'> kw;
  typedef ExistentialFormula type;

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_existential_preconditions()) {
      throw pegtl::parse_error("using 'exists' formula without enabling "
                               ":existential-preconditions requirement",
                               in.current_position());
    }
  }
};

template <> struct Existential<Effect> {
  typedef keyword<'e', 'x', 'i', 's', 't', 's'> kw;
  typedef ExistentialEffect type;

  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_conditional_effects()) {
      throw pegtl::parse_error("using 'exists' effect without enabling "
                               ":conditional-effects requirement",
                               in.current_position());
    }
  }
};

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule>
struct open_quantification
    : pegtl::action<
          action,
          pegtl::seq<
              pegtl::one<'('>, ignored,
              typename QuantificationOperator<typename QuantificationProxy<
                  QuantificationOperator, QuantifiedRule>::BasePDDLType>::kw,
              ignored, pegtl::one<'('>, ignored>> {
  typedef QuantificationProxy<QuantificationOperator, QuantifiedRule> QP;
  typedef QuantificationOperator<typename QP::BasePDDLType> QO;
};

template <typename Rule> struct open_quantification_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::QP QP;
    QP::parsing_stack(s).push(std::make_shared<typename Rule::QO::type>());
    typedef typename Rule::QO QO;
    QO::check_requirement(in, s);
  }
};

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule>
struct forall_variable : pegtl::action<action, name> {
  typedef QuantificationProxy<QuantificationOperator, QuantifiedRule> QP;
  typedef QuantificationOperator<typename QP::BasePDDLType> QO;
};

template <typename Rule> struct forall_variable_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::QP QP;
    s.name = in.string();
    s.variable_list.push_back(std::static_pointer_cast<typename Rule::QO::type>(
                                  QP::parsing_stack(s).top())
                                  ->append_variable(s.name));
    auto i = s.registered_variables.insert(std::make_pair(
        '?' + StringConverter::tolower(s.name), s.variable_list.back()));
    if (!i.second) {
      throw pegtl::parse_error(
          "variable '?" + s.name +
              "' already existing in the current parsing context",
          in.current_position());
    }
  }
};

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule>
struct close_quantification
    : pegtl::action<
          action,
          pegtl::seq<
              typed_var_list<pegtl::action<
                  forall_variable_action,
                  forall_variable<QuantificationOperator, QuantifiedRule>>>,
              ignored, pegtl::one<')'>, ignored, QuantifiedRule, ignored,
              pegtl::one<')'>>> {
  typedef QuantificationProxy<QuantificationOperator, QuantifiedRule> QP;
  typedef QuantificationOperator<typename QP::BasePDDLType> QO;
};

template <typename Rule> struct close_quantification_action {
  static void apply0(state &s) {
    typedef typename Rule::QP QP;
    auto ct = std::static_pointer_cast<typename Rule::QO::type>(
        QP::parsing_stack(s).top());
    QP::set_quantified(ct, QP::last_parsed(s));
    for (const auto &v : ct->get_variables()) {
      s.registered_variables.erase(v->get_name());
    }
    QP::last_parsed(s) = QP::parsing_stack(s).top();
    QP::parsing_stack(s).pop();
  }
};

template <template <typename> class QuantificationOperator,
          typename QuantifiedRule>
struct quantification
    : pegtl::if_must<pegtl::action<open_quantification_action,
                                   open_quantification<QuantificationOperator,
                                                       QuantifiedRule>>,
                     pegtl::action<close_quantification_action,
                                   close_quantification<QuantificationOperator,
                                                        QuantifiedRule>>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_QUANTIFICATION_HH
