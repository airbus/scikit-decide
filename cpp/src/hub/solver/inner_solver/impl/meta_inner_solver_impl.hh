/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_META_INNER_SOLVER_IMPL_HH
#define SKDECIDE_META_INNER_SOLVER_IMPL_HH

namespace skdecide {

#define SK_META_INNER_SOLVER_TEMPLATE_DECL                                     \
  template <typename TSolver, typename TDomain>

#define SK_META_INNER_SOLVER_CLASS MetaInnerSolver<TSolver, TDomain>

SK_META_INNER_SOLVER_TEMPLATE_DECL
SK_META_INNER_SOLVER_CLASS::MetaInnerSolver(std::unique_ptr<TSolver> solver)
    : _solver(std::move(solver)) {}

SK_META_INNER_SOLVER_TEMPLATE_DECL
void SK_META_INNER_SOLVER_CLASS::solve(const State &s) { _solver->solve(s); }

SK_META_INNER_SOLVER_TEMPLATE_DECL
void SK_META_INNER_SOLVER_CLASS::clear() { _solver->clear(); }

SK_META_INNER_SOLVER_TEMPLATE_DECL
bool SK_META_INNER_SOLVER_CLASS::is_solution_defined_for(const State &s) {
  return _solver->is_solution_defined_for(s);
}

SK_META_INNER_SOLVER_TEMPLATE_DECL
const typename SK_META_INNER_SOLVER_CLASS::Action &
SK_META_INNER_SOLVER_CLASS::get_best_action(const State &s) {
  return _solver->get_best_action(s);
}

SK_META_INNER_SOLVER_TEMPLATE_DECL
typename SK_META_INNER_SOLVER_CLASS::Value
SK_META_INNER_SOLVER_CLASS::get_best_value(const State &s) {
  return _solver->get_best_value(s);
}

SK_META_INNER_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_META_INNER_SOLVER_CLASS::State>::Set
SK_META_INNER_SOLVER_CLASS::get_explored_states() const {
  return _solver->get_explored_states();
}

} // namespace skdecide

#endif // SKDECIDE_META_INNER_SOLVER_IMPL_HH
