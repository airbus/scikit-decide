/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSPREPLAN_HH
#define SKDECIDE_SSPREPLAN_HH

#include <chrono>
#include <functional>
#include <memory>
#include <utility>

#include "hub/solver/inner_solver/meta_inner_solver_base.hh"
#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy,
          typename TdeterminizationAdapter>
class SSPReplanSolver {
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

  using Plan = std::vector<std::pair<State, Action>>;

  typedef std::function<bool(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<bool(const SSPReplanSolver &, Domain &)>
      CallbackFunctor;

  /**
   * @brief SSP-Replan solver: determinizes a stochastic domain and replans
   * whenever the actual outcome deviates from the expected one.
   *
   * @param domain The stochastic domain to solve.
   * @param adapter Determinization adapter that wraps the stochastic domain
   *   into a deterministic interface for the inner solver.
   * @param factory Factory function that creates an inner deterministic solver
   *   given a reference to the determinized domain.
   * @param goal_checker Functor returning true when a state is a goal.
   * @param max_replans Maximum number of replanning episodes before giving up.
   *   Defaults to 1000.
   * @param max_steps Maximum total simulation steps across all replans.
   *   Defaults to 10000.
   * @param callback Functor called after each replan; return true to stop
   *   early. Defaults to never stop.
   * @param verbose Whether to log progress messages. Defaults to false.
   */
  SSPReplanSolver(
      Domain &domain, Adapter adapter, InnerSolverFactory factory,
      const GoalCheckerFunctor &goal_checker, std::size_t max_replans = 1000,
      std::size_t max_steps = 10000,
      const CallbackFunctor &callback = [](const SSPReplanSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);
  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s);
  double get_best_value(const State &s) const;

  const Plan &get_plan() const;
  std::size_t get_nb_replans() const;
  std::size_t get_nb_steps() const;
  std::size_t get_solving_time() const;
  double get_total_cost() const;

private:
  Domain &_domain;
  Adapter _adapter;
  InnerSolverFactory _factory;
  GoalCheckerFunctor _goal_checker;
  CallbackFunctor _callback;
  std::size_t _max_replans;
  std::size_t _max_steps;
  bool _verbose;

  void _replan(const State &s);

  using PolicyMap =
      typename MapTypeDeducer<State, std::pair<Action, double>>::Map;
  PolicyMap _policy;

  std::unique_ptr<InnerSolver> _inner_solver;
  bool _has_solution = false;
  bool _adapter_initialized = false;
  State _expected_current;

  Plan _current_plan;
  std::size_t _nb_replans = 0;
  std::size_t _nb_steps = 0;
  std::size_t _solving_time = 0;
  double _total_cost = 0.0;
};

} // namespace skdecide

#endif // SKDECIDE_SSPREPLAN_HH
