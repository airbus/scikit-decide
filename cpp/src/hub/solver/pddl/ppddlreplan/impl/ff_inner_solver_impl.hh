/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FF_INNER_SOLVER_IMPL_HH
#define SKDECIDE_FF_INNER_SOLVER_IMPL_HH

#include "hub/solver/pddl/ppddlreplan/ff_inner_solver.hh"

#include <exception>
#include <string>

#include "utils/logging.hh"

namespace skdecide {

namespace pddl {

#define SK_FF_INNER_SOLVER_TEMPLATE_DECL template <typename Texec>

#define SK_FF_INNER_SOLVER_CLASS FFInnerSolver<Texec>

SK_FF_INNER_SOLVER_TEMPLATE_DECL
std::unique_ptr<SK_FF_INNER_SOLVER_CLASS>
SK_FF_INNER_SOLVER_CLASS::create_from_params(PddlDeterministicDomain &domain,
                                             const InnerSolverParams &params,
                                             bool verbose) {
  double dead_end_cost = params.get<double>("dead_end_cost", 1e9);
  return std::make_unique<FFInnerSolver<Texec>>(domain.task(), dead_end_cost,
                                                verbose);
}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
SK_FF_INNER_SOLVER_CLASS::FFInnerSolver(const Task &task, double dead_end_cost,
                                        bool verbose)
    : _ff(
          task, dead_end_cost, [](const FFSolver<Texec> &) { return false; },
          verbose),
      _solve_succeeded(false) {}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
void SK_FF_INNER_SOLVER_CLASS::solve(const PddlState &s) {
  _solve_succeeded = false;
  try {
    _ff.solve(s);
    _solve_succeeded = true;
  } catch (const std::exception &e) {
    Logger::warn(std::string("FFInnerSolver: inner planner failed: ") +
                 e.what());
  }
}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
void SK_FF_INNER_SOLVER_CLASS::clear() {
  _ff.clear();
  _solve_succeeded = false;
}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
bool SK_FF_INNER_SOLVER_CLASS::is_solution_defined_for(const PddlState &s) {
  return _ff.is_solution_defined_for(s);
}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
const PddlAction &
SK_FF_INNER_SOLVER_CLASS::get_best_action(const PddlState &s) {
  return static_cast<const PddlAction &>(_ff.get_best_action(s));
}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
PddlValue SK_FF_INNER_SOLVER_CLASS::get_best_value(const PddlState &) {
  return PddlValue(0.0);
}

SK_FF_INNER_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_FF_INNER_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<PddlState>::Set result;
  for (const auto &s : _ff.get_explored_states()) {
    result.insert(PddlState(s));
  }
  return result;
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_FF_INNER_SOLVER_IMPL_HH
