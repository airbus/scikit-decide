/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_SUCCESSOR_GENERATOR_HH
#define SKDECIDE_PDDL_SEMANTICS_SUCCESSOR_GENERATOR_HH

#include <vector>

#include "applicable_actions_generator.hh"
#include "state.hh"

namespace skdecide {

namespace pddl {

class Task;

struct Successor {
  State state;
  double probability;
};

class SuccessorGenerator {
public:
  SuccessorGenerator(const Task &task);

  std::vector<Successor> get_successors(const State &state,
                                        const GroundAction &action) const;

private:
  const Task &_task;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_SUCCESSOR_GENERATOR_HH
