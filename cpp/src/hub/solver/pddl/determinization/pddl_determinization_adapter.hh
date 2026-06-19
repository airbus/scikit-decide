/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DETERMINIZATION_ADAPTER_HH
#define SKDECIDE_PDDL_DETERMINIZATION_ADAPTER_HH

#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "hub/domain/pddl/semantics/determinize_effects.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "hub/solver/pddl/pddl_domain_adapter.hh"
#include "hub/solver/determinization/determinized_domain.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief Determinization adapter that converts a stochastic PPDDL task into
 * a deterministic PDDL task by resolving probabilistic effects.
 *
 * The strategy tag selects how stochastic effects are resolved:
 * - AllOutcomesStrategy: each probabilistic branch becomes a separate
 *   deterministic action.
 * - MostProbableOutcomeStrategy: only the highest-probability branch is kept.
 * - RandomOutcomeStrategy: one branch is sampled at random (re-sampled on
 *   each call to update()).
 *
 * Used by PPDDLReplanSolver, PPDDLPlanMergerSolver, and
 * PPDDLDetHindsightSolver to feed a determinized domain to the inner solver.
 *
 * @tparam TstrategyTag One of AllOutcomesStrategy,
 *         MostProbableOutcomeStrategy, or RandomOutcomeStrategy.
 * @tparam Texec Execution policy (SequentialExecution or ParallelExecution).
 */
template <typename TstrategyTag, typename Texec>
class PddlEffectDeterminizationAdapter {
public:
  using DeterminizedDomainType = PddlDeterministicDomain;
  using State = PddlState;
  using Action = PddlAction;
  using DetAction = PddlAction;

  /**
   * @param task Original stochastic PPDDL task. The adapter creates an
   *        internal determinized copy according to TstrategyTag.
   */
  PddlEffectDeterminizationAdapter(const Task &task);

  void update();
  DeterminizedDomainType &domain();
  const Task &determinized_task() const { return *_det_task; }
  PddlAction to_original(const PddlAction &det_action) const;
  PddlState expected_next(const PddlState &s, const PddlAction &a);
  bool needs_update_each_replan() const;

private:
  DeterminizationMode strategy_mode() const;

  const Task &_original_task;
  std::unique_ptr<Task> _det_task;
  std::vector<ActionMapping> _action_mapping;
  std::unique_ptr<PddlDeterministicDomain> _det_domain;
  mutable std::mt19937 _rng{std::random_device{}()};
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DETERMINIZATION_ADAPTER_HH
