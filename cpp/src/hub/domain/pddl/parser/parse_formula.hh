/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSE_FORMULA_HH
#define SKDECIDE_PDDL_PARSE_FORMULA_HH

#include "pegtl.hpp"

#include "parser_state.hh"
#include "parser_action.hh"

#include "parse_predicate_head.hh"
#include "parse_negation.hh"
#include "parse_aggregation.hh"
#include "parse_quantification.hh"
#include "parse_implication.hh"
#include "parse_comparison.hh"
#include "parse_constraints.hh"
#include "parse_preference.hh"
#include "parse_terms_equality.hh"

namespace pegtl = TAO_PEGTL_NAMESPACE; // NOLINT

namespace skdecide {

namespace pddl {

namespace parser {

struct formula
    : pegtl::sor<
          negation<formula>, aggregation<Conjunction, formula>,
          aggregation<Disjunction, formula>, quantification<Universal, formula>,
          quantification<Existential, formula>, implication<formula>,
          terms_equality, // before comparison to disambiguate the '=' operator
          comparison, predicate_head<formula>> {};

struct timed_formula : pegtl::sor<unary_constraint<AtStartOperator>,
                                  unary_constraint<AtEndOperator>,
                                  unary_constraint<OverAllOperator>> {};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSE_FORMULA_HH
