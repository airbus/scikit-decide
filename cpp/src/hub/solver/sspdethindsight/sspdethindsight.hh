/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSPDETHINDSIGHT_HH
#define SKDECIDE_SSPDETHINDSIGHT_HH

#include <chrono>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "hub/solver/inner_solver/meta_inner_solver_base.hh"
#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy,
          typename TdeterminizationAdapter>
class SSPDetHindsightSolver {
public:
  typedef Tdomain Domain;
  typedef typename Domain::State State;
  typedef typename Domain::Action Action;
  typedef typename Domain::Value Value;
  typedef Texecution_policy ExecutionPolicy;

  typedef TdeterminizationAdapter Adapter;
  typedef typename Adapter::DeterminizedDomainType DetDomain;
  typedef typename DetDomain::Action DetAction;

  using InnerSolver = MetaInnerSolverBase<DetDomain>;
  using InnerSolverFactory =
      std::function<std::unique_ptr<InnerSolver>(DetDomain &)>;
  using AdapterFactory = std::function<Adapter()>;

  typedef std::function<bool(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<bool(const SSPDetHindsightSolver &, Domain &)>
      CallbackFunctor;

  /**
   * @brief Hindsight optimization solver for stochastic shortest path (SSP)
   * domains. At each state, samples multiple determinized scenarios, solves
   * each with an inner deterministic solver, and selects the action with the
   * lowest average cost across scenarios.
   *
   * @param domain The stochastic domain to solve.
   * @param inner_factory Factory that creates an inner deterministic solver
   *   given a reference to the determinized domain.
   * @param adapter_factory Factory that creates a fresh determinization
   *   adapter instance for each hindsight scenario.
   * @param goal_checker Functor returning true when a state is a goal.
   * @param sample_width Number of random determinization scenarios to sample
   *   at each decision point. Defaults to 30.
   * @param dead_end_cost Cost penalty assigned when the inner solver cannot
   *   find a plan from a successor state. Defaults to 1000.0.
   * @param max_steps Maximum total simulation steps. Defaults to 10000.
   * @param discount Discount factor applied to future costs when averaging
   *   across scenarios. Defaults to 0.99.
   * @param epsilon Convergence threshold used for value comparisons.
   *   Defaults to 1e-3.
   * @param callback Functor called after each hindsight evaluation; return
   *   true to stop early. Defaults to never stop.
   * @param verbose Whether to log progress messages. Defaults to false.
   */
  SSPDetHindsightSolver(
      Domain &domain, InnerSolverFactory inner_factory,
      AdapterFactory adapter_factory, const GoalCheckerFunctor &goal_checker,
      std::size_t sample_width = 30, double dead_end_cost = 1000.0,
      std::size_t max_steps = 10000, double discount = 0.99,
      double epsilon = 1e-3,
      const CallbackFunctor &callback = [](const SSPDetHindsightSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);
  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s);
  double get_best_value(const State &s) const;

  std::size_t get_nb_steps() const;
  std::size_t get_solving_time() const;

  using StateSet = typename SetTypeDeducer<State>::Set;
  const StateSet &get_explored_states() const;
  const StateSet &get_terminal_states() const;

private:
  Domain &_domain;
  InnerSolverFactory _inner_factory;
  AdapterFactory _adapter_factory;
  GoalCheckerFunctor _goal_checker;
  CallbackFunctor _callback;
  std::size_t _sample_width;
  double _dead_end_cost;
  std::size_t _max_steps;
  double _discount;
  double _epsilon;
  bool _verbose;

  void _evaluate_hindsight(const State &s);

  mutable Texecution_policy _execution_policy;

  using PolicyMap =
      typename MapTypeDeducer<State, std::pair<Action, double>>::Map;
  PolicyMap _policy;

  bool _has_solution = false;
  std::size_t _nb_steps = 0;
  std::size_t _solving_time = 0;

  StateSet _explored_states;
  StateSet _terminal_states;
};

} // namespace skdecide

#endif // SKDECIDE_SSPDETHINDSIGHT_HH
