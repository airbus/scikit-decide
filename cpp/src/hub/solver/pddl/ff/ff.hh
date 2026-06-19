/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FF_HH
#define SKDECIDE_PDDL_FF_HH

#include <functional>
#include <memory>
#include <vector>

#include "hub/domain/pddl/heuristics/ff_heuristic.hh"
#include "hub/domain/pddl/semantics/goal_checker.hh"
#include "hub/domain/pddl/semantics/state.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "hub/solver/ehc/ehc.hh"
#include "ff_domain_adapter.hh"
#include "utils/execution.hh"

namespace skdecide {

namespace pddl {

/**
 * FF (Fast-Forward) planning solver.
 *
 * Uses Enforced Hill Climbing (EHC) with the h_FF heuristic and helpful
 * actions as described in:
 *
 *   Hoffmann, J. (2001). FF: The Fast-Forward Planning System.
 *   AI Magazine, 22(3), 57-62.
 */
template <typename Texecution_policy = SequentialExecution> class FFSolver {
public:
  typedef Texecution_policy ExecutionPolicy;
  typedef std::function<bool(const FFSolver &)> CallbackFunctor;

  /**
   * @param task Parsed PDDL task providing actions, initial state, and goal.
   * @param dead_end_cost Cost assigned to states where FF finds no plan.
   *        Defaults to 1e9.
   * @param callback Called after each EHC iteration; return true to stop.
   * @param verbose Enable progress logging.
   */
  FFSolver(
      const Task &task, double dead_end_cost = 1e9,
      const CallbackFunctor &callback = [](const FFSolver &) { return false; },
      bool verbose = false);

  void solve(const State &initial_state);
  void clear();
  bool is_solution_defined_for(const State &s) const;
  const GroundAction &get_best_action(const State &s) const;
  std::vector<std::pair<State, GroundAction>> get_plan() const;
  std::size_t get_nb_explored_states() const;
  std::vector<State> get_explored_states() const;
  std::size_t get_solving_time() const;

private:
  const Task &_task;
  double _dead_end_cost;
  CallbackFunctor _callback;
  bool _verbose;

  std::unique_ptr<FFHeuristic> _heuristic;
  std::unique_ptr<PddlDeterministicDomain> _domain;
  std::unique_ptr<GoalChecker> _goal_checker;
  std::unique_ptr<EHCSolver<PddlDeterministicDomain, Texecution_policy>> _ehc;

  mutable PddlState _last_state;
  mutable std::pair<double, std::vector<GroundAction>> _last_result;
  mutable bool _has_cached = false;

  PddlState _initial_state;

  void ensure_computed(const PddlState &s) const;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_FF_HH
