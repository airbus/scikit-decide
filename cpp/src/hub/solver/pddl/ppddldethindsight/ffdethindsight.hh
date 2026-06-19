/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FFDETHINDSIGHT_HH
#define SKDECIDE_FFDETHINDSIGHT_HH

#include <functional>
#include <memory>

#include "hub/domain/pddl/semantics/state.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "ppddldethindsight.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief FF-DetHindsight: hindsight optimization for PPDDL domains using FF
 * as the inner deterministic planner.
 *
 * Thin wrapper around PPDDLDetHindsightSolver with inner_solver_name fixed
 * to "FF". Counterpart of FFReplan.
 *
 * Reference: Yoon, S. W., Fern, A., & Givan, R. (2008). Probabilistic Planning
 * via Determinization in Hindsight. In Proc. AAAI, pp. 1010-1016.
 *
 * @tparam Texecution_policy Execution policy (Sequential or Parallel)
 */
template <typename Texecution_policy> class FFDetHindsightSolver {
public:
  typedef std::function<bool(const FFDetHindsightSolver &)> CallbackFunctor;

  /**
   * @param task Parsed PPDDL task (stochastic effects).
   * @param sample_width Number of random determinization scenarios sampled
   *        per action at each step. Defaults to 30.
   * @param dead_end_cost Cost penalty for dead-end states where FF fails.
   *        Defaults to 1e9.
   * @param max_steps Maximum total simulation steps. Defaults to 10000.
   * @param discount Discount factor for value evaluation (< 1 for convergence
   *        with dead-end terminals). Defaults to 0.99.
   * @param epsilon Convergence threshold for value evaluation. Defaults to
   *        1e-3.
   * @param callback Called after each hindsight evaluation; return true to
   *        stop.
   * @param verbose Enable progress logging.
   */
  FFDetHindsightSolver(
      const Task &task, std::size_t sample_width = 30,
      double dead_end_cost = 1e9, std::size_t max_steps = 10000,
      double discount = 0.99, double epsilon = 1e-3,
      const CallbackFunctor &callback =
          [](const FFDetHindsightSolver &) { return false; },
      bool verbose = false);

  void solve(const State &s);
  void clear();
  bool is_solution_defined_for(const State &s) const;
  const GroundAction &get_best_action(const State &s);

  double get_best_value(const State &s) const;
  std::size_t get_nb_steps() const;
  std::size_t get_solving_time() const;

  typename SetTypeDeducer<PddlState>::Set get_explored_states() const;
  typename SetTypeDeducer<PddlState>::Set get_terminal_states() const;

private:
  CallbackFunctor _callback_storage;
  std::unique_ptr<PPDDLDetHindsightSolver<Texecution_policy>> _impl;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_FFDETHINDSIGHT_HH
