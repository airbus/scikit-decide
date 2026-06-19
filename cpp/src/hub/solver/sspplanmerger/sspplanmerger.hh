/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSPPLANMERGER_HH
#define SKDECIDE_SSPPLANMERGER_HH

#include <chrono>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "hub/solver/inner_solver/meta_inner_solver_base.hh"
#include "utils/associative_container_deducer.hh"
#include "utils/execution.hh"

namespace skdecide {

template <typename Tdomain, typename Texecution_policy,
          typename TdeterminizationAdapter>
class SSPPlanMergerSolver {
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
  using PolicyMap =
      typename MapTypeDeducer<State, std::pair<Action, double>>::Map;

  typedef std::function<bool(Domain &, const State &)> GoalCheckerFunctor;
  typedef std::function<bool(const SSPPlanMergerSolver &, Domain &)>
      CallbackFunctor;

  /**
   * @brief SSP plan-merging solver: iteratively determinizes a stochastic
   * domain, plans from terminal states, and merges plans into a policy graph
   * until Monte-Carlo assessment shows the replanning probability is below
   * a threshold (rho).
   *
   * @param domain The stochastic domain to solve.
   * @param adapter_factory Factory that creates a fresh determinization
   *   adapter instance for each planning episode.
   * @param inner_factory Factory that creates an inner deterministic solver
   *   given a reference to the determinized domain.
   * @param goal_checker Functor returning true when a state is a goal.
   * @param rho Replanning probability threshold: the algorithm stops when
   *   MC evaluation estimates the probability of leaving the policy graph
   *   is below rho. Defaults to 0.1.
   * @param mc_samples Number of Monte-Carlo rollout samples used to estimate
   *   the replanning probability each iteration. Defaults to 100.
   * @param max_iterations Maximum number of plan-merge iterations.
   *   Defaults to 50.
   * @param max_steps Maximum simulation steps per MC rollout.
   *   Defaults to 10000.
   * @param dead_end_cost Cost assigned to terminal states that are not goals
   *   (dead-ends) during policy-graph optimization. Defaults to 1e9.
   * @param optimize_policy_graph Whether to run discounted value iteration on
   *   the merged policy graph after each planning round, treating non-goal
   *   terminals as absorbing dead-ends. Defaults to false.
   * @param discount Discount factor used during policy-graph optimization
   *   (must be < 1 for convergence when dead-end terminals exist).
   *   Defaults to 0.99.
   * @param epsilon Convergence threshold for value iteration on the policy
   *   graph. Defaults to 1e-3.
   * @param callback Functor called after each iteration; return true to stop
   *   early. Defaults to never stop.
   * @param verbose Whether to log progress messages. Defaults to false.
   */
  SSPPlanMergerSolver(
      Domain &domain, AdapterFactory adapter_factory,
      InnerSolverFactory inner_factory, const GoalCheckerFunctor &goal_checker,
      double rho = 0.1, std::size_t mc_samples = 100,
      std::size_t max_iterations = 50, std::size_t max_steps = 10000,
      double dead_end_cost = 1e9, bool optimize_policy_graph = false,
      double discount = 0.99, double epsilon = 1e-3,
      const CallbackFunctor &callback = [](const SSPPlanMergerSolver &,
                                           Domain &) { return false; },
      bool verbose = false);

  void clear();
  void solve(const State &s);
  void resolve(const State &s);
  bool is_solution_defined_for(const State &s) const;
  const Action &get_best_action(const State &s);
  double get_best_value(const State &s) const;

  std::size_t get_nb_iterations() const;
  std::size_t get_nb_plans() const;
  std::size_t get_solving_time() const;
  std::size_t get_policy_size() const;

  typename SetTypeDeducer<State>::Set get_explored_states() const;
  typename SetTypeDeducer<State>::Set get_terminal_states() const;
  PolicyMap get_policy() const;

private:
  Domain &_domain;
  AdapterFactory _adapter_factory;
  InnerSolverFactory _inner_factory;
  GoalCheckerFunctor _goal_checker;
  CallbackFunctor _callback;
  double _rho;
  std::size_t _mc_samples;
  std::size_t _max_iterations;
  std::size_t _max_steps;
  double _dead_end_cost;
  bool _optimize_policy_graph;
  double _discount;
  double _epsilon;
  bool _verbose;

  void _plan_from(const State &s);
  void _plan_from_terminals(const std::vector<State> &terminals);
  void _optimize_ssp(const State &s0);
  void _evaluate_policy() const;
  State _sample_successor(const State &s, const Action &a);

  mutable PolicyMap _policy;
  mutable bool _values_evaluated = false;

  bool _has_solution = false;
  std::size_t _nb_iterations = 0;
  std::size_t _nb_plans = 0;
  std::size_t _solving_time = 0;

  mutable std::mt19937 _rng{std::random_device{}()};
};

} // namespace skdecide

#endif // SKDECIDE_SSPPLANMERGER_HH
