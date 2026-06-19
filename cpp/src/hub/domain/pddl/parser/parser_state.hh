/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_PARSER_STATE_HH
#define SKDECIDE_PDDL_PARSER_STATE_HH

#include <list>
#include <stack>
#include <unordered_map>

#include "domain.hh"
#include "predicate.hh"
#include "function.hh"
#include "formula.hh"
#include "effect.hh"
#include "expression.hh"
#include "class.hh"
#include "number.hh"
#include "operator.hh"
#include "derived_predicate.hh"
#include "problem.hh"

namespace skdecide {

namespace pddl {

namespace parser {

// parsing state

struct state {

  std::string name; // current parsed name

  Domain::Ptr domain; // last parsed PDDL domain
  std::unordered_map<std::string, Domain::Ptr>
      domains;                    // list of parsed domains
  Requirements::Ptr requirements; // current parsed requirements
  Requirements::Ptr
      global_requirements; // domain and problem requirements altogether
  std::list<Domain::TypePtr> type_list;
  std::list<Domain::ObjectPtr> object_list;
  std::list<Predicate::VariablePtr>
      variable_list; // get from Predicate but would be the same for fluents or
                     // actions
  Domain::PredicatePtr predicate; // current parsed predicate
  Domain::FunctionPtr function;   // current parsed function
  std::unordered_map<std::string, Domain::ObjectPtr> registered_objects;
  std::unordered_map<std::string, Predicate::VariablePtr>
      registered_variables; // get from Predicate but would be the same for
                            // fluents or actions
  std::stack<Formula::Ptr> formulas;       // current nested parsed formulas
  Formula::Ptr formula;                    // last parsed formula
  std::stack<Effect::Ptr> effects;         // current nested parsed effects
  Effect::Ptr effect;                      // last parsed effect
  std::stack<Expression::Ptr> expressions; // current nested parsed expressions
  Expression::Ptr expression;              // last parsed expression
  std::stack<Number::Ptr> numbers;         // current nested parsed numbers
  Number::Ptr number;                      // last parsed number
  Class::Ptr cls;                          // last parsed class
  Action::Ptr action;                      // last parsed action
  DurativeAction::Ptr durative_action;     // last parsed durative action
  Event::Ptr event;                        // last parsed event
  Process::Ptr process;                    // last parsed process

  Problem::Ptr problem; // last parsed PDDL problem
  std::unordered_map<std::string, Problem::Ptr>
      problems; // list of parsed problems
};

} // namespace parser

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PARSER_STATE_HH
