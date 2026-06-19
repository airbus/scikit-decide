/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_AGGREGATION_HH
#define SKDECIDE_PDDL_PARSE_AGGREGATION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "aggregation_formula.hh"
#include "aggregation_effect.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct constraint_goal;
struct formula;
struct timed_precondition;
struct pref_con_goal;
struct c_effect;
struct p_effect;
struct a_effect_da;
struct p_effect_da;
struct process_effect;
struct duration_constraint;
template <typename FormulaRule> struct precondition_preference;

template <template <typename> class AggregationOperator,
          typename AggregatedRule, typename Enable = void>
struct AggregationProxy;

template <template <typename> class AggregationOperator,
          typename AggregatedRule>
struct AggregationProxy<
    AggregationOperator, AggregatedRule,
    typename std::enable_if<
        std::is_same<AggregatedRule, constraint_goal>::value ||
        std::is_same<AggregatedRule, formula>::value ||
        std::is_same<AggregatedRule, precondition_preference<formula>>::value ||
        std::is_same<AggregatedRule, timed_precondition>::value ||
        std::is_same<AggregatedRule, pref_con_goal>::value ||
        std::is_same<AggregatedRule, duration_constraint>::value>::type> {
  typedef Formula BasePDDLType;
  typedef typename AggregationOperator<BasePDDLType>::type DerivedPDDLType;
  static std::stack<Formula::Ptr> &parsing_stack(state &s) {
    return s.formulas;
  }
  static Formula::Ptr &last_parsed(state &s) { return s.formula; }
  static void append_aggregated(typename DerivedPDDLType::Ptr d,
                                Formula::Ptr f) {
    d->append_formula(f);
  }
};

template <template <typename> class AggregationOperator,
          typename AggregatedRule>
struct AggregationProxy<
    AggregationOperator, AggregatedRule,
    typename std::enable_if<
        std::is_same<AggregatedRule, c_effect>::value ||
        std::is_same<AggregatedRule, p_effect>::value ||
        std::is_same<AggregatedRule, a_effect_da>::value ||
        std::is_same<AggregatedRule, p_effect_da>::value ||
        std::is_same<AggregatedRule, process_effect>::value>::type> {
  typedef Effect BasePDDLType;
  typedef typename AggregationOperator<BasePDDLType>::type DerivedPDDLType;
  static std::stack<Effect::Ptr> &parsing_stack(state &s) { return s.effects; }
  static Effect::Ptr &last_parsed(state &s) { return s.effect; }
  static void append_aggregated(typename DerivedPDDLType::Ptr d,
                                Effect::Ptr e) {
    d->append_effect(e);
  }
};

template <typename BasePDDLType> struct Conjunction;

template <> struct Conjunction<Formula> {
  typedef keyword<'a', 'n', 'd'> kw;
  typedef ConjunctionFormula type;
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {}
};

template <> struct Conjunction<Effect> {
  typedef keyword<'a', 'n', 'd'> kw;
  typedef ConjunctionEffect type;
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {}
};

template <typename BasePDDLType> struct Disjunction;

template <> struct Disjunction<Formula> {
  typedef keyword<'o', 'r'> kw;
  typedef DisjunctionFormula type;
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {
    if (!s.global_requirements->has_disjunctive_preconditions()) {
      throw pegtl::parse_error("using disjunctive formula without enabling "
                               ":disjunctive-preconditions requirement",
                               in.current_position());
    }
  }
};

template <> struct Disjunction<Effect> {
  typedef keyword<'o', 'n', 'e', 'o', 'f'> kw;
  typedef DisjunctionEffect type;
  template <typename Input>
  static void check_requirement(const Input &in, state &s) {}
};

template <template <typename> class AggregationOperator,
          typename AggregatedRule>
struct open_aggregation
    : pegtl::action<
          action,
          pegtl::seq<
              pegtl::one<'('>, ignored,
              typename AggregationOperator<typename AggregationProxy<
                  AggregationOperator, AggregatedRule>::BasePDDLType>::kw,
              ignored>> {
  typedef AggregationProxy<AggregationOperator, AggregatedRule> AP;
  typedef AggregationOperator<typename AP::BasePDDLType> AO;
};

template <typename Rule> struct open_aggregation_action {
  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::AP AP;
    typedef typename Rule::AO AO;
    AO::check_requirement(in, s);
    AP::parsing_stack(s).push(std::make_shared<typename AO::type>());
  }
};

template <template <typename> class AggregationOperator,
          typename AggregatedRule>
struct aggregated : pegtl::action<action, AggregatedRule> {
  typedef AggregationProxy<AggregationOperator, AggregatedRule> AP;
  typedef AggregationOperator<typename AP::BasePDDLType> AO;
};

template <typename Rule> struct aggregated_action {
  static void apply0(state &s) {
    typedef typename Rule::AP AP;
    AP::append_aggregated(std::static_pointer_cast<typename Rule::AO::type>(
                              AP::parsing_stack(s).top()),
                          AP::last_parsed(s));
  }
};

template <template <typename> class AggregationOperator,
          typename AggregatedRule>
struct close_aggregation
    : pegtl::action<
          action,
          pegtl::seq<pegtl::list<pegtl::action<aggregated_action,
                                               aggregated<AggregationOperator,
                                                          AggregatedRule>>,
                                 ignored>,
                     ignored, pegtl::one<')'>>> {
  typedef AggregationProxy<AggregationOperator, AggregatedRule> AP;
};

template <typename Rule> struct close_aggregation_action {
  static void apply0(state &s) {
    typedef typename Rule::AP AP;
    AP::last_parsed(s) = AP::parsing_stack(s).top();
    AP::parsing_stack(s).pop();
  }
};

template <template <typename> class AggregationOperator,
          typename AggregatedRule>
struct aggregation
    : pegtl::if_must<
          pegtl::action<open_aggregation_action,
                        open_aggregation<AggregationOperator, AggregatedRule>>,
          pegtl::action<
              close_aggregation_action,
              close_aggregation<AggregationOperator, AggregatedRule>>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_AGGREGATION_HH
