/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DETERMINIZATION_ADAPTER_IMPL_HH
#define SKDECIDE_PDDL_DETERMINIZATION_ADAPTER_IMPL_HH

#include "hub/solver/pddl/determinization/pddl_determinization_adapter.hh"

namespace skdecide {

namespace pddl {

#define SK_PDDL_DET_ADAPTER_TEMPLATE_DECL                                      \
  template <typename TstrategyTag, typename Texec>

#define SK_PDDL_DET_ADAPTER_CLASS                                              \
  PddlEffectDeterminizationAdapter<TstrategyTag, Texec>

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
SK_PDDL_DET_ADAPTER_CLASS::PddlEffectDeterminizationAdapter(const Task &task)
    : _original_task(task) {}

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
void SK_PDDL_DET_ADAPTER_CLASS::update() {
  bool all_outcomes = std::is_same_v<TstrategyTag, AllOutcomesStrategy>;
  DeterminizationMode mode = strategy_mode();
  auto det =
      determinize_actions(_original_task.actions(), mode, all_outcomes, _rng);
  _det_task = std::make_unique<Task>(_original_task, std::move(det.actions));
  _action_mapping = std::move(det.mapping);
  _det_domain = std::make_unique<PddlDeterministicDomain>(*_det_task);
}

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
typename SK_PDDL_DET_ADAPTER_CLASS::DeterminizedDomainType &
SK_PDDL_DET_ADAPTER_CLASS::domain() {
  return *_det_domain;
}

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
PddlAction
SK_PDDL_DET_ADAPTER_CLASS::to_original(const PddlAction &det_action) const {
  auto &m = _action_mapping[det_action.action_id];
  PddlAction orig = det_action;
  orig.action_id = m.original_action_id;
  return orig;
}

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
PddlState SK_PDDL_DET_ADAPTER_CLASS::expected_next(const PddlState &s,
                                                   const PddlAction &a) {
  return _det_domain->get_next_state(s, a);
}

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
bool SK_PDDL_DET_ADAPTER_CLASS::needs_update_each_replan() const {
  return std::is_same_v<TstrategyTag, RandomOutcomeStrategy>;
}

SK_PDDL_DET_ADAPTER_TEMPLATE_DECL
DeterminizationMode SK_PDDL_DET_ADAPTER_CLASS::strategy_mode() const {
  if constexpr (std::is_same_v<TstrategyTag, MostProbableOutcomeStrategy>) {
    return DeterminizationMode::MostProbable;
  } else {
    return DeterminizationMode::Random;
  }
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DETERMINIZATION_ADAPTER_IMPL_HH
