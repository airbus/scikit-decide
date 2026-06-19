/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_META_INNER_SOLVER_PROXY_IMPL_HH
#define SKDECIDE_META_INNER_SOLVER_PROXY_IMPL_HH

namespace skdecide {

#define SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL                               \
  template <typename Tdomain, typename Texecution_policy>

#define SK_META_INNER_SOLVER_PROXY_CLASS                                       \
  MetaInnerSolverProxy<Tdomain, Texecution_policy>

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
SK_META_INNER_SOLVER_PROXY_CLASS::MetaInnerSolverProxy(
    Domain &domain, GoalCheckerFunctor goal_checker, HeuristicFunctor heuristic,
    SspFactory factory)
    : _impl(factory(domain, std::move(goal_checker), std::move(heuristic))) {}

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
SK_META_INNER_SOLVER_PROXY_CLASS::MetaInnerSolverProxy(
    Domain &domain, GoalCheckerFunctor goal_checker, HeuristicFunctor heuristic,
    TerminalValueFunctor terminal_value, FretFactory factory)
    : _impl(factory(domain, std::move(goal_checker), std::move(heuristic),
                    std::move(terminal_value))) {}

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
void SK_META_INNER_SOLVER_PROXY_CLASS::solve(const State &s) {
  _impl->solve(s);
}

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
void SK_META_INNER_SOLVER_PROXY_CLASS::clear() { _impl->clear(); }

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
bool SK_META_INNER_SOLVER_PROXY_CLASS::is_solution_defined_for(const State &s) {
  return _impl->is_solution_defined_for(s);
}

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
const typename SK_META_INNER_SOLVER_PROXY_CLASS::Action &
SK_META_INNER_SOLVER_PROXY_CLASS::get_best_action(const State &s) {
  return _impl->get_best_action(s);
}

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
typename SK_META_INNER_SOLVER_PROXY_CLASS::Value
SK_META_INNER_SOLVER_PROXY_CLASS::get_best_value(const State &s) {
  return _impl->get_best_value(s);
}

SK_META_INNER_SOLVER_PROXY_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_META_INNER_SOLVER_PROXY_CLASS::State>::Set
SK_META_INNER_SOLVER_PROXY_CLASS::get_explored_states() const {
  return _impl->get_explored_states();
}

} // namespace skdecide

#endif // SKDECIDE_META_INNER_SOLVER_PROXY_IMPL_HH
