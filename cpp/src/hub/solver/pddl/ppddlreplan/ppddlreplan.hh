/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PPDDLREPLAN_HH
#define SKDECIDE_PPDDLREPLAN_HH

#include <functional>
#include <memory>

#include "hub/domain/pddl/semantics/goal_checker.hh"
#include "hub/domain/pddl/semantics/state.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "hub/solver/sspreplan/sspreplan.hh"

#include "hub/solver/pddl/determinization/pddl_determinization_adapter.hh"
#include "hub/solver/pddl/determinization/pddl_domain_for_determinization.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief PPDDL-Replan: reactive replanning for probabilistic PDDL (PPDDL)
 * domains with a pluggable inner deterministic solver.
 *
 * Repeatedly determinizes the stochastic domain at the PDDL effect-tree level,
 * plans with the selected inner solver, executes the plan, and replans when the
 * actual outcome deviates from the expected deterministic outcome.
 *
 * Counterpart of FFReplan, but supports pluggable inner solvers via name.
 *
 * Reference: Yoon, S. W., Fern, A., & Givan, R. (2007). FF-Replan: A Baseline
 * for Probabilistic Planning. In Proc. ICAPS, pp. 352-359.
 *
 * @tparam Texecution_policy Execution policy (Sequential or Parallel)
 * @tparam TdeterminizationStrategy Determinization strategy tag
 */
template <typename Texecution_policy, typename TdeterminizationStrategy>
class PPDDLReplanSolver {
public:
  using Adapter = PddlEffectDeterminizationAdapter<TdeterminizationStrategy,
                                                   Texecution_policy>;
  using Domain = PddlDomainForDeterminization;
  using Solver = SSPReplanSolver<Domain, Texecution_policy, Adapter>;
  using InnerSolverFactory = typename Solver::InnerSolverFactory;

  typedef std::function<bool(const PPDDLReplanSolver &)> CallbackFunctor;

  /**
   * @param task Parsed PPDDL task (stochastic effects).
   * @param inner_solver_factory Factory that creates the inner deterministic
   *        solver given a PddlDeterministicDomain and solver params.
   * @param max_replans Maximum number of replanning episodes before giving up.
   *        Defaults to 1000.
   * @param max_steps Maximum total simulation steps across all episodes.
   *        Defaults to 10000.
   * @param callback Called after each replan; return true to stop.
   * @param verbose Enable progress logging.
   */
  PPDDLReplanSolver(
      const Task &task, InnerSolverFactory inner_solver_factory,
      std::size_t max_replans = 1000, std::size_t max_steps = 10000,
      const CallbackFunctor &callback =
          [](const PPDDLReplanSolver &) { return false; },
      bool verbose = false);

  void solve(const State &s);
  void clear();
  bool is_solution_defined_for(const State &s) const;
  const GroundAction &get_best_action(const State &s);

  std::vector<std::pair<State, GroundAction>> get_plan() const;
  std::size_t get_nb_replans() const;
  std::size_t get_nb_steps() const;
  std::size_t get_solving_time() const;
  double get_total_cost() const;

private:
  std::unique_ptr<Domain> _stochastic_domain;
  std::unique_ptr<Solver> _solver;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PPDDLREPLAN_HH
