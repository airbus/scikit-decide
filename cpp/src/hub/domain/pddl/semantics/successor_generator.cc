/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "successor_generator.hh"
#include "task.hh"

#include "../operator.hh"

namespace skdecide {

namespace pddl {

SuccessorGenerator::SuccessorGenerator(const Task &task) : _task(task) {}

std::vector<Successor>
SuccessorGenerator::get_successors(const State &state,
                                   const GroundAction &action) const {
  auto &effect = _task.actions()[action.action_id]->get_effect();
  if (!effect) {
    return {{state.copy(), 1.0}};
  }

  auto outcomes = effect->apply(state, _task, action.binding);
  std::vector<Successor> result;
  result.reserve(outcomes.size());
  for (auto &[prob, s] : outcomes) {
    result.push_back({std::move(s), prob});
  }
  return result;
}

} // namespace pddl

} // namespace skdecide
