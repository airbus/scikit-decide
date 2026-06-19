/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "goal_checker.hh"
#include "task.hh"

#include "../formula.hh"

namespace skdecide {

namespace pddl {

GoalChecker::GoalChecker(const Task &task) : _task(task) {}

bool GoalChecker::is_goal(const State &state) const {
  auto &goal = _task.goal();
  if (!goal) {
    return false;
  }
  Binding empty_binding;
  return goal->holds(state, _task, empty_binding);
}

} // namespace pddl

} // namespace skdecide
