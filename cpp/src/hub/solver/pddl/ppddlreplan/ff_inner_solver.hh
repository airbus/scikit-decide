/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FF_INNER_SOLVER_HH
#define SKDECIDE_FF_INNER_SOLVER_HH

#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/meta_inner_solver_base.hh"
#include "hub/solver/pddl/ff/ff.hh"
#include "hub/solver/pddl/pddl_domain_adapter.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief Inner solver adapter that wraps FFSolver for use inside meta-solvers
 * (SSPReplan, SSPPlanMerger, SSPDetHindsight).
 *
 * Conforms to the MetaInnerSolverBase interface so it can be plugged into
 * the inner solver registry and instantiated by name ("FF").
 *
 * @tparam Texec Execution policy (SequentialExecution or ParallelExecution).
 */
template <typename Texec>
class FFInnerSolver : public MetaInnerSolverBase<PddlDeterministicDomain> {
public:
  /**
   * @param task Parsed PDDL task providing actions, initial state, and goal.
   * @param dead_end_cost Cost assigned to states where FF finds no plan.
   * @param verbose Enable progress logging.
   */
  FFInnerSolver(const Task &task, double dead_end_cost, bool verbose);

  static std::unique_ptr<FFInnerSolver<Texec>>
  create_from_params(PddlDeterministicDomain &domain,
                     const InnerSolverParams &params, bool verbose);

  void solve(const PddlState &s) override;
  void clear() override;
  bool is_solution_defined_for(const PddlState &s) override;
  const PddlAction &get_best_action(const PddlState &s) override;
  PddlValue get_best_value(const PddlState &s) override;
  typename SetTypeDeducer<PddlState>::Set get_explored_states() const override;

private:
  FFSolver<Texec> _ff;
  bool _solve_succeeded = false;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_FF_INNER_SOLVER_HH
