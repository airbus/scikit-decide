/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_HH
#define SKDECIDE_PDDL_HH

#include <list>

#include "domain.hh"
#include "type.hh"
#include "object.hh"
#include "variable.hh"
#include "term.hh"
#include "predicate.hh"
#include "function.hh"
#include "derived_predicate.hh"
#include "class.hh"
#include "operator.hh"
#include "constraint_formula.hh"
#include "preference.hh"
#include "predicate_formula.hh"
#include "equality_formula.hh"
#include "quantified_formula.hh"
#include "aggregation_formula.hh"
#include "imply_formula.hh"
#include "negation_formula.hh"
#include "timed_formula.hh"
#include "duration_formula.hh"
#include "comparison_formula.hh"
#include "operation_expression.hh"
#include "minus_expression.hh"
#include "numerical_expression.hh"
#include "function_expression.hh"
#include "timed_expression.hh"
#include "duration_expression.hh"
#include "predicate_effect.hh"
#include "aggregation_effect.hh"
#include "quantified_effect.hh"
#include "conditional_effect.hh"
#include "negation_effect.hh"
#include "timed_effect.hh"
#include "duration_effect.hh"
#include "assignment_effect.hh"
#include "problem.hh"
#include "optimization_expression.hh"
#include "totaltime_expression.hh"
#include "totalcost_expression.hh"
#include "violation_expression.hh"
#include "probabilistic_effect.hh"
#include "reward_expression.hh"
#include "goal_achieved_expression.hh"

namespace skdecide {

namespace pddl {

class PDDL {
public:
  /**
   * Constructs a PDDL object (domain and problem) from PDDL files
   * @param @param files List of files containing domain and problem
   * descriptions
   * @param verbose Activates parsing traces
   */
  void load(const std::list<std::string> &files, bool verbose = false);

  const std::list<Domain::Ptr> &get_domains();
  const std::list<Problem::Ptr> &get_problems();

private:
  std::list<Domain::Ptr> _domains;
  std::list<Problem::Ptr> _problems;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_HH
