/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FFREPLAN_HH
#define SKDECIDE_FFREPLAN_HH

#include <functional>
#include <memory>

#include "hub/domain/pddl/semantics/state.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "ppddlreplan.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief FF-Replan from Yoon, Fern & Givan (ICAPS 2007): reactive replanning
 * for probabilistic PDDL (PPDDL) domains.
 *
 * Thin wrapper around PPDDLReplanSolver with the inner solver fixed to FF.
 *
 * Reference: Yoon, S. W., Fern, A., & Givan, R. (2007). FF-Replan: A Baseline
 * for Probabilistic Planning. In Proc. ICAPS, pp. 352-359.
 *
 * @tparam Texecution_policy Execution policy (Sequential or Parallel)
 * @tparam TdeterminizationStrategy Determinization strategy tag
 *   (AllOutcomesStrategy, MostProbableOutcomeStrategy, or
 * RandomOutcomeStrategy)
 */
template <typename Texecution_policy, typename TdeterminizationStrategy>
class FFReplanSolver {
public:
  using PPDDL = PPDDLReplanSolver<Texecution_policy, TdeterminizationStrategy>;

  typedef std::function<bool(const FFReplanSolver &)> CallbackFunctor;

  /**
   * @param task Parsed PPDDL task (stochastic effects).
   * @param dead_end_cost Cost penalty for dead-end states where FF fails.
   *        Defaults to 1e9.
   * @param max_replans Maximum number of replanning episodes before giving up.
   *        Defaults to 1000.
   * @param max_steps Maximum total simulation steps across all episodes.
   *        Defaults to 10000.
   * @param callback Called after each replan; return true to stop.
   * @param verbose Enable progress logging.
   */
  FFReplanSolver(
      const Task &task, double dead_end_cost = 1e9,
      std::size_t max_replans = 1000, std::size_t max_steps = 10000,
      const CallbackFunctor &callback =
          [](const FFReplanSolver &) { return false; },
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
  CallbackFunctor _callback_storage;
  std::unique_ptr<PPDDL> _impl;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_FFREPLAN_HH
