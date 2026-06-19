/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PPDDLPLANMERGER_HH
#define SKDECIDE_PPDDLPLANMERGER_HH

#include <functional>
#include <memory>

#include "hub/domain/pddl/semantics/goal_checker.hh"
#include "hub/domain/pddl/semantics/state.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "hub/solver/sspplanmerger/sspplanmerger.hh"

#include "hub/solver/pddl/determinization/pddl_determinization_adapter.hh"
#include "hub/solver/pddl/determinization/pddl_domain_for_determinization.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief Plan-merging solver for probabilistic PDDL (PPDDL) domains with a
 * pluggable inner deterministic solver.
 *
 * Implements the RFF plan-aggregation algorithm using PDDL effect-tree
 * determinization. Iteratively determinizes the domain, plans from terminal
 * states, and merges plans into a policy until the Monte-Carlo replanning
 * probability drops below threshold rho.
 *
 * Reference: Teichteil-Königsbuch, F., Kuter, U., & Infantes, G. (2010).
 * RFF: A Robust FF-Based MDP Planning Algorithm for Generating Policies with
 * Low Probability of Failure. In Proc. AAMAS, pp. 801-808.
 *
 * @tparam Texecution_policy Execution policy (Sequential or Parallel)
 * @tparam TdeterminizationStrategy Determinization strategy tag
 */
template <typename Texecution_policy, typename TdeterminizationStrategy>
class PPDDLPlanMergerSolver {
public:
  using Adapter = PddlEffectDeterminizationAdapter<TdeterminizationStrategy,
                                                   Texecution_policy>;
  using Domain = PddlDomainForDeterminization;
  using Solver = SSPPlanMergerSolver<Domain, Texecution_policy, Adapter>;
  using InnerSolverFactory = typename Solver::InnerSolverFactory;

  typedef std::function<bool(const PPDDLPlanMergerSolver &)> CallbackFunctor;

  /**
   * @param task Parsed PPDDL task (stochastic effects).
   * @param inner_solver_factory Factory that creates the inner deterministic
   *        solver given a PddlDeterministicDomain and solver params.
   * @param rho Replanning probability threshold; iteration stops when MC
   *        estimated replan probability drops below rho. Defaults to 0.1.
   * @param mc_samples Number of Monte-Carlo rollout samples per iteration.
   *        Defaults to 100.
   * @param max_iterations Maximum plan-merge iterations. Defaults to 50.
   * @param max_steps Maximum steps per MC rollout. Defaults to 10000.
   * @param dead_end_cost Cost penalty for dead-end terminal states.
   *        Defaults to 1e9.
   * @param optimize_policy_graph If true, run discounted value iteration on
   *        the policy graph after each merge to optimize action choices.
   *        Defaults to false.
   * @param discount Discount factor for policy-graph VI (< 1 for convergence
   *        with dead-end terminals). Defaults to 0.99.
   * @param epsilon Convergence threshold for VI residual. Defaults to 1e-3.
   * @param callback Called after each iteration; return true to stop.
   * @param verbose Enable progress logging.
   */
  PPDDLPlanMergerSolver(
      const Task &task, InnerSolverFactory inner_solver_factory,
      double rho = 0.1, std::size_t mc_samples = 100,
      std::size_t max_iterations = 50, std::size_t max_steps = 10000,
      double dead_end_cost = 1e9, bool optimize_policy_graph = false,
      double discount = 0.99, double epsilon = 1e-3,
      const CallbackFunctor &callback =
          [](const PPDDLPlanMergerSolver &) { return false; },
      bool verbose = false);

  void solve(const State &s);
  void resolve(const State &s);
  void clear();
  bool is_solution_defined_for(const State &s) const;
  const GroundAction &get_best_action(const State &s) const;
  double get_best_value(const State &s) const;

  std::size_t get_nb_iterations() const;
  std::size_t get_nb_plans() const;
  std::size_t get_solving_time() const;
  std::size_t get_policy_size() const;

  typename Solver::PolicyMap get_policy() const;
  typename SetTypeDeducer<PddlState>::Set get_explored_states() const;
  typename SetTypeDeducer<PddlState>::Set get_terminal_states() const;

private:
  const Task &_task;
  std::unique_ptr<Domain> _stochastic_domain;
  std::unique_ptr<Solver> _solver;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PPDDLPLANMERGER_HH
