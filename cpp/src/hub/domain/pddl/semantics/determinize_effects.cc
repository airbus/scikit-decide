/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "determinize_effects.hh"

namespace skdecide {

namespace pddl {

DeterminizedActions determinize_actions(
    const std::vector<std::shared_ptr<Action>> &original_actions,
    DeterminizationMode mode, bool all_outcomes, std::mt19937 &rng) {
  DeterminizedActions result;

  for (int i = 0; i < static_cast<int>(original_actions.size()); ++i) {
    auto &action = original_actions[i];
    auto &effect = action->get_effect();

    if (all_outcomes && effect) {
      auto dets = effect->all_determinizations(effect, rng);
      for (int j = 0; j < static_cast<int>(dets.size()); ++j) {
        auto det_action = std::make_shared<Action>(*action);
        det_action->set_effect(dets[j]);
        result.actions.push_back(std::move(det_action));
        result.mapping.push_back({i, j});
      }
    } else if (effect) {
      auto det_effect = effect->determinize(effect, mode, rng);
      auto det_action = std::make_shared<Action>(*action);
      det_action->set_effect(det_effect);
      result.actions.push_back(std::move(det_action));
      result.mapping.push_back({i, -1});
    } else {
      result.actions.push_back(std::make_shared<Action>(*action));
      result.mapping.push_back({i, -1});
    }
  }

  return result;
}

} // namespace pddl

} // namespace skdecide
