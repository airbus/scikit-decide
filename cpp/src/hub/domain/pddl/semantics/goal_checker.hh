/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_GOAL_CHECKER_HH
#define SKDECIDE_PDDL_SEMANTICS_GOAL_CHECKER_HH

#include "state.hh"

namespace skdecide {

namespace pddl {

class Task;

class GoalChecker {
public:
  GoalChecker(const Task &task);

  bool is_goal(const State &state) const;

private:
  const Task &_task;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_GOAL_CHECKER_HH
