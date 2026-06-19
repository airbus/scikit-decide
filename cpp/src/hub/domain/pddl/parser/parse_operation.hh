/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_OPERATION_HH
#define SKDECIDE_PDDL_PARSE_OPERATION_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "operation_expression.hh"
#include "minus_expression.hh"
#include "comparison_formula.hh"
#include "assignment_effect.hh"
#include "parse_function_head.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct expression;
struct ground_f_exp;
struct timed_expression;
struct number_expression;
struct durative_expression;
struct duration_expression;

struct AddOperator {
  typedef pegtl::one<'+'> kw;
  typedef AddExpression type;
};

struct SubOperator {
  typedef pegtl::one<'-'> kw;
  typedef MinusExpression
      type; // we first try to match a unary minus expression
};

struct MulOperator {
  typedef pegtl::one<'*'> kw;
  typedef MulExpression type;
};

struct DivOperator {
  typedef pegtl::one<'/'> kw;
  typedef DivExpression type;
};

struct GreaterOperator {
  typedef pegtl::one<'>'> kw;
  typedef GreaterFormula type;
};

struct GreaterEqOperator {
  typedef pegtl::string<'>', '='> kw;
  typedef GreaterEqFormula type;
};

struct LessOperator {
  typedef pegtl::one<'<'> kw;
  typedef LessFormula type;
};

struct LessEqOperator {
  typedef pegtl::string<'<', '='> kw;
  typedef LessEqFormula type;
};

struct EqOperator {
  typedef pegtl::one<'='> kw;
  typedef EqFormula type;
};

struct AssignOperator {
  typedef keyword<'a', 's', 's', 'i', 'g', 'n'> kw;
  typedef AssignEffect type;
};

struct IncreaseOperator {
  typedef keyword<'i', 'n', 'c', 'r', 'e', 'a', 's', 'e'> kw;
  typedef IncreaseEffect type;
};

struct DecreaseOperator {
  typedef keyword<'d', 'e', 'c', 'r', 'e', 'a', 's', 'e'> kw;
  typedef DecreaseEffect type;
};

struct ScaleUpOperator {
  typedef keyword<'s', 'c', 'a', 'l', 'e', '-', 'u', 'p'> kw;
  typedef ScaleUpEffect type;
};

struct ScaleDownOperator {
  typedef keyword<'s', 'c', 'a', 'l', 'e', '-', 'd', 'o', 'w', 'n'> kw;
  typedef ScaleDownEffect type;
};

struct AssignInitOperator {
  typedef pegtl::one<'='> kw;
  typedef AssignEffect type;
};

template <typename Operator, typename Enable = void>
struct OperationExpressionProxy;

template <typename Operator>
struct OperationExpressionProxy<
    Operator,
    typename std::enable_if<std::is_same<Operator, AddOperator>::value ||
                            std::is_same<Operator, MulOperator>::value ||
                            std::is_same<Operator, DivOperator>::value>::type> {
  static std::stack<Expression::Ptr> &parsing_stack(state &s) {
    return s.expressions;
  }
  static Expression::Ptr &last_parsed(state &s) { return s.expression; }
  static void set_lhs(Expression::Ptr &e, const Expression::Ptr &l) {
    std::static_pointer_cast<typename Operator::type>(e)->set_left_expression(
        l);
  }
  static void set_rhs(Expression::Ptr &e, const Expression::Ptr &r) {
    std::static_pointer_cast<typename Operator::type>(e)->set_right_expression(
        r);
  }
};

template <typename Operator>
struct OperationExpressionProxy<
    Operator,
    typename std::enable_if<std::is_same<Operator, SubOperator>::value>::type> {
  static std::stack<Expression::Ptr> &parsing_stack(state &s) {
    return s.expressions;
  }
  static Expression::Ptr &last_parsed(state &s) { return s.expression; }
  static void set_lhs(Expression::Ptr &e, const Expression::Ptr &l) {
    std::static_pointer_cast<typename Operator::type>(e)->set_expression(l);
  }
  static void set_rhs(Expression::Ptr &e, const Expression::Ptr &r) {
    Expression::Ptr left_exp =
        std::static_pointer_cast<typename Operator::type>(e)->get_expression();
    e = std::make_shared<SubExpression>(left_exp, r);
  }
};

template <typename Operator>
struct OperationExpressionProxy<
    Operator,
    typename std::enable_if<std::is_same<Operator, GreaterOperator>::value ||
                            std::is_same<Operator, GreaterEqOperator>::value ||
                            std::is_same<Operator, LessOperator>::value ||
                            std::is_same<Operator, LessEqOperator>::value ||
                            std::is_same<Operator, EqOperator>::value>::type> {
  static std::stack<Formula::Ptr> &parsing_stack(state &s) {
    return s.formulas;
  }
  static Formula::Ptr &last_parsed(state &s) { return s.formula; }
  static void set_lhs(Formula::Ptr &f, const Expression::Ptr &l) {
    std::static_pointer_cast<typename Operator::type>(f)->set_left_expression(
        l);
  }
  static void set_rhs(Formula::Ptr &f, const Expression::Ptr &r) {
    std::static_pointer_cast<typename Operator::type>(f)->set_right_expression(
        r);
  }
};

template <typename Operator>
struct OperationExpressionProxy<
    Operator, typename std::enable_if<
                  std::is_same<Operator, AssignOperator>::value ||
                  std::is_same<Operator, IncreaseOperator>::value ||
                  std::is_same<Operator, DecreaseOperator>::value ||
                  std::is_same<Operator, ScaleUpOperator>::value ||
                  std::is_same<Operator, ScaleDownOperator>::value ||
                  std::is_same<Operator, AssignInitOperator>::value>::type> {
  static std::stack<Effect::Ptr> &parsing_stack(state &s) { return s.effects; }
  static Effect::Ptr &last_parsed(state &s) { return s.effect; }
  static void set_lhs(Effect::Ptr &e, const Expression::Ptr &f) {
    std::static_pointer_cast<typename Operator::type>(e)->set_function(
        std::static_pointer_cast<FunctionExpression>(f));
  }
  static void set_rhs(Effect::Ptr &e, const Expression::Ptr &ee) {
    std::static_pointer_cast<typename Operator::type>(e)->set_expression(ee);
  }
};

template <typename Operator, typename ExpressionRule>
struct open_operation_expression
    : pegtl::action<action, pegtl::seq<pegtl::one<'('>, ignored,
                                       typename Operator::kw, ignored>> {
  typedef OperationExpressionProxy<Operator> OEP;
  typedef Operator OP;
  typedef ExpressionRule ER;
};

template <typename Rule> struct open_op_exp_action {
  template <typename Operator, typename ExpressionRule, typename Enable = void>
  struct check_requirement;

  template <typename Operator, typename ExpressionRule>
  struct check_requirement<
      Operator, ExpressionRule,
      typename std::enable_if<
          std::is_same<ExpressionRule, expression>::value ||
          std::is_same<ExpressionRule, timed_expression>::value ||
          std::is_same<ExpressionRule, durative_expression>::value ||
          std::is_same<ExpressionRule, number_expression>::value ||
          std::is_same<ExpressionRule, ground_f_exp>::value>::type> {
    template <typename Input> static void perform(const Input &in, state &s) {
      if (!s.global_requirements->has_numeric_fluents()) {
        throw pegtl::parse_error("using expression operator without enabling "
                                 ":numeric-fluents requirement",
                                 in.current_position());
      }
    }
  };

  template <typename Operator, typename ExpressionRule>
  struct check_requirement<
      Operator, ExpressionRule,
      typename std::enable_if<
          std::is_same<ExpressionRule, duration_expression>::value &&
          (std::is_same<Operator, LessEqOperator>::value ||
           std::is_same<Operator, GreaterEqOperator>::value)>::type> {
    template <typename Input> static void perform(const Input &in, state &s) {
      if (!s.global_requirements->has_duration_inequalities()) {
        throw pegtl::parse_error("using duration expression inequality without "
                                 "enabling :duration-inequalities requirement",
                                 in.current_position());
      }
    }
  };

  template <typename Operator, typename ExpressionRule>
  struct check_requirement<
      Operator, ExpressionRule,
      typename std::enable_if<
          std::is_same<ExpressionRule, duration_expression>::value &&
          std::is_same<Operator, EqOperator>::value>::type> {
    template <typename Input> static void perform(const Input &in, state &s) {}
  };

  template <typename Input> static void apply(const Input &in, state &s) {
    typedef typename Rule::OEP OEP;
    typedef typename Rule::OP OP;
    typedef typename Rule::ER ER;
    check_requirement<OP, ER>::perform(in, s);
    OEP::parsing_stack(s).push(std::make_shared<typename OP::type>());
  }
};

template <typename Operator, typename ExpressionRule>
struct operation_lhs : pegtl::action<action, ExpressionRule> {
  typedef OperationExpressionProxy<Operator> OEP;
};

template <typename Rule> struct operation_lhs_action {
  static void apply0(state &s) {
    typedef typename Rule::OEP OEP;
    OEP::set_lhs(OEP::parsing_stack(s).top(), s.expression);
  }
};

template <typename Operator, typename ExpressionRule>
struct operation_rhs : pegtl::action<action, ExpressionRule> {
  typedef OperationExpressionProxy<Operator> OEP;
};

template <typename Rule> struct operation_rhs_action {
  static void apply0(state &s) {
    typedef typename Rule::OEP OEP;
    OEP::set_rhs(OEP::parsing_stack(s).top(), s.expression);
  }
};

template <typename Operator, typename ExpressionRule>
struct close_operation_expression
    : pegtl::action<
          action,
          pegtl::seq<
              pegtl::action<
                  operation_lhs_action,
                  operation_lhs<
                      Operator,
                      typename std::conditional<
                          std::is_same<Operator, AssignOperator>::value ||
                              std::is_same<Operator, IncreaseOperator>::value ||
                              std::is_same<Operator, DecreaseOperator>::value ||
                              std::is_same<Operator, ScaleUpOperator>::value ||
                              std::is_same<Operator,
                                           ScaleDownOperator>::value ||
                              std::is_same<Operator, AssignInitOperator>::value,
                          function_head, ExpressionRule>::type>>,
              ignored,
              typename std::conditional<
                  std::is_same<ExpressionRule, duration_expression>::value,
                  pegtl::action<operation_rhs_action,
                                operation_rhs<Operator, expression>>,
                  typename std::conditional<
                      std::is_same<Operator, SubOperator>::value,
                      pegtl::opt<pegtl::action<
                          operation_rhs_action,
                          operation_rhs<Operator, ExpressionRule>>>,
                      pegtl::action<operation_rhs_action,
                                    operation_rhs<Operator, ExpressionRule>>>::
                      type>::type,
              ignored, pegtl::one<')'>>> {
  typedef OperationExpressionProxy<Operator> OEP;
};

template <typename Rule> struct close_op_exp_action {
  static void apply0(state &s) {
    typedef typename Rule::OEP OEP;
    OEP::last_parsed(s) = OEP::parsing_stack(s).top();
    OEP::parsing_stack(s).pop();
  }
};

template <typename Operator, typename ExpressionRule>
struct operation_expression
    : pegtl::if_must<
          pegtl::action<open_op_exp_action,
                        open_operation_expression<Operator, ExpressionRule>>,
          pegtl::action<close_op_exp_action,
                        close_operation_expression<Operator, ExpressionRule>>> {
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_OPERATION_HH
