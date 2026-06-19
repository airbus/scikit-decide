/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "../semantics/task.hh"

#include "../comparison_formula.hh"
#include "../duration_expression.hh"
#include "../function_expression.hh"
#include "../goal_achieved_expression.hh"
#include "../minus_expression.hh"
#include "../numerical_expression.hh"
#include "../operation_expression.hh"
#include "../optimization_expression.hh"
#include "../reward_expression.hh"
#include "../timed_expression.hh"
#include "../totalcost_expression.hh"
#include "../totaltime_expression.hh"
#include "../violation_expression.hh"

// Include template implementations for explicit instantiation
#include "binary_expression_impl.hh"

#include <stdexcept>

namespace skdecide {

namespace pddl {

// === print() implementations ===

std::ostream &GoalAchievedExpression::print(std::ostream &o) const {
  o << "goal-achieved";
  return o;
}

std::ostream &RewardExpression::print(std::ostream &o) const {
  o << "reward";
  return o;
}

std::ostream &TotalCostExpression::print(std::ostream &o) const {
  o << "total-cost";
  return o;
}

std::ostream &TotalTimeExpression::print(std::ostream &o) const {
  o << "total-time";
  return o;
}

std::ostream &TimeExpression::print(std::ostream &o) const {
  o << "#t";
  return o;
}

std::ostream &ViolationExpression::print(std::ostream &o) const {
  o << "(is-violated " << *_preference << ")";
  return o;
}

void ViolationExpression::set_preference(const Preference::Ptr &preference) {
  _preference = preference;
}

// === evaluate() implementations ===

// --- NumericalExpression ---
double NumericalExpression::evaluate(const State & /*state*/,
                                     const Task & /*task*/,
                                     const Binding & /*binding*/) const {
  return get_number()->as_double();
}

// --- FunctionExpression ---
double FunctionExpression::evaluate(const State &state, const Task &task,
                                    const Binding &binding) const {
  int fid = task.function_id(get_function()->get_name());
  GroundTuple args;
  for (auto &t : get_terms()) {
    args.push_back(task.resolve_term(t, binding));
  }
  auto &fmap = state.fluents[fid];
  auto it = fmap.find(args);
  if (it != fmap.end()) {
    return it->second;
  }
  return 0.0;
}

// --- AddExpression ---
double AddExpression::evaluate(const State &state, const Task &task,
                               const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) +
         get_right_expression()->evaluate(state, task, binding);
}

// --- SubExpression ---
double SubExpression::evaluate(const State &state, const Task &task,
                               const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) -
         get_right_expression()->evaluate(state, task, binding);
}

// --- MulExpression ---
double MulExpression::evaluate(const State &state, const Task &task,
                               const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) *
         get_right_expression()->evaluate(state, task, binding);
}

// --- DivExpression ---
double DivExpression::evaluate(const State &state, const Task &task,
                               const Binding &binding) const {
  return get_left_expression()->evaluate(state, task, binding) /
         get_right_expression()->evaluate(state, task, binding);
}

// --- MinusExpression ---
double MinusExpression::evaluate(const State &state, const Task &task,
                                 const Binding &binding) const {
  return -get_expression()->evaluate(state, task, binding);
}

// --- MinimizeExpression ---
double MinimizeExpression::evaluate(const State &state, const Task &task,
                                    const Binding &binding) const {
  return get_expression()->evaluate(state, task, binding);
}

// --- MaximizeExpression ---
double MaximizeExpression::evaluate(const State &state, const Task &task,
                                    const Binding &binding) const {
  return get_expression()->evaluate(state, task, binding);
}

// --- TotalCostExpression ---
double TotalCostExpression::evaluate(const State &state, const Task &task,
                                     const Binding & /*binding*/) const {
  int fid = task.total_cost_function();
  if (fid < 0) {
    return 0.0;
  }
  auto &fmap = state.fluents[fid];
  GroundTuple empty_args;
  auto it = fmap.find(empty_args);
  if (it != fmap.end()) {
    return it->second;
  }
  return 0.0;
}

// --- TotalTimeExpression ---
double TotalTimeExpression::evaluate(const State &state, const Task & /*task*/,
                                     const Binding & /*binding*/) const {
  return state.time;
}

// --- TimeExpression ---
double TimeExpression::evaluate(const State &state, const Task & /*task*/,
                                const Binding & /*binding*/) const {
  return state.dt;
}

// --- DurationExpression ---
double DurationExpression::evaluate(const State &state, const Task & /*task*/,
                                    const Binding & /*binding*/) const {
  return state.duration;
}

// --- RewardExpression ---
double RewardExpression::evaluate(const State &state, const Task &task,
                                  const Binding & /*binding*/) const {
  int fid = task.reward_function();
  if (fid < 0) {
    return 0.0;
  }
  auto &fmap = state.fluents[fid];
  GroundTuple empty_args;
  auto it = fmap.find(empty_args);
  if (it != fmap.end()) {
    return it->second;
  }
  return 0.0;
}

// --- GoalAchievedExpression ---
double GoalAchievedExpression::evaluate(const State &state, const Task &task,
                                        const Binding &binding) const {
  return task.goal()->holds(state, task, binding) ? 1.0 : 0.0;
}

// --- ViolationExpression ---
double ViolationExpression::evaluate(const State & /*state*/,
                                     const Task & /*task*/,
                                     const Binding & /*binding*/) const {
  return 0.0;
}

// --- Comparison formulas also inherit from Expression via BinaryExpression ---
// They need evaluate() but it doesn't make semantic sense (they're boolean).
double GreaterFormula::evaluate(const State & /*state*/, const Task & /*task*/,
                                const Binding & /*binding*/) const {
  throw std::runtime_error(
      "GreaterFormula cannot be evaluated as numeric expression");
}

double GreaterEqFormula::evaluate(const State & /*state*/,
                                  const Task & /*task*/,
                                  const Binding & /*binding*/) const {
  throw std::runtime_error(
      "GreaterEqFormula cannot be evaluated as numeric expression");
}

double LessFormula::evaluate(const State & /*state*/, const Task & /*task*/,
                             const Binding & /*binding*/) const {
  throw std::runtime_error(
      "LessFormula cannot be evaluated as numeric expression");
}

double LessEqFormula::evaluate(const State & /*state*/, const Task & /*task*/,
                               const Binding & /*binding*/) const {
  throw std::runtime_error(
      "LessEqFormula cannot be evaluated as numeric expression");
}

double EqFormula::evaluate(const State & /*state*/, const Task & /*task*/,
                           const Binding & /*binding*/) const {
  throw std::runtime_error(
      "EqFormula cannot be evaluated as numeric expression");
}

} // namespace pddl

} // namespace skdecide
