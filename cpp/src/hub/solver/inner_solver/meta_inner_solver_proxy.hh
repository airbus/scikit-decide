/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_META_INNER_SOLVER_PROXY_HH
#define SKDECIDE_META_INNER_SOLVER_PROXY_HH

#include <functional>
#include <memory>

#include "meta_inner_solver_base.hh"
#include "utils/associative_container_deducer.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy>
class MetaInnerSolverProxy {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef typename Domain::Predicate Predicate;
  typedef Texecution_policy ExecutionPolicy;

  typedef std::function<Predicate(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<Value(Domain &, const State &)> HeuristicFunctor;
  typedef std::function<Value(const State &)> TerminalValueFunctor;

  using SspFactory = std::function<std::unique_ptr<MetaInnerSolverBase<Domain>>(
      Domain &, GoalCheckerFunctor, HeuristicFunctor)>;

  using FretFactory =
      std::function<std::unique_ptr<MetaInnerSolverBase<Domain>>(
          Domain &, GoalCheckerFunctor, HeuristicFunctor,
          TerminalValueFunctor)>;

  // SSiPP constructor: (domain, gc, h, ssp_factory)
  MetaInnerSolverProxy(Domain &domain, GoalCheckerFunctor goal_checker,
                       HeuristicFunctor heuristic, SspFactory factory);

  // FRET constructor: (domain, gc, h, terminal_value, fret_factory)
  MetaInnerSolverProxy(Domain &domain, GoalCheckerFunctor goal_checker,
                       HeuristicFunctor heuristic,
                       TerminalValueFunctor terminal_value,
                       FretFactory factory);

  void solve(const State &s);
  void clear();
  bool is_solution_defined_for(const State &s);
  const Action &get_best_action(const State &s);
  Value get_best_value(const State &s);
  typename SetTypeDeducer<State>::Set get_explored_states() const;

private:
  std::unique_ptr<MetaInnerSolverBase<Domain>> _impl;
};

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/meta_inner_solver_proxy_impl.hh"
#endif

#endif // SKDECIDE_META_INNER_SOLVER_PROXY_HH
