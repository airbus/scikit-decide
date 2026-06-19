/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_DETERMINIZE_EFFECTS_HH
#define SKDECIDE_PDDL_SEMANTICS_DETERMINIZE_EFFECTS_HH

#include <memory>
#include <random>
#include <vector>

#include "../effect.hh"
#include "../operator.hh"

namespace skdecide {

namespace pddl {

struct ActionMapping {
  int original_action_id;
  int outcome_index; // -1 for non-probabilistic actions
};

struct DeterminizedActions {
  std::vector<std::shared_ptr<Action>> actions;
  std::vector<ActionMapping> mapping;
};

DeterminizedActions determinize_actions(
    const std::vector<std::shared_ptr<Action>> &original_actions,
    DeterminizationMode mode, bool all_outcomes, std::mt19937 &rng);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_DETERMINIZE_EFFECTS_HH
