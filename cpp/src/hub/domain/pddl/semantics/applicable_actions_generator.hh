/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEMANTICS_APPLICABLE_ACTIONS_GENERATOR_HH
#define SKDECIDE_PDDL_SEMANTICS_APPLICABLE_ACTIONS_GENERATOR_HH

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "state.hh"

namespace Clingo {
class Control;
class Symbol;
} // namespace Clingo

namespace skdecide {

namespace pddl {

class Task;

struct GroundAction {
  int action_id;
  Binding binding;
  std::vector<int> arguments;

  bool operator==(const GroundAction &other) const {
    return action_id == other.action_id && arguments == other.arguments;
  }

  std::size_t hash() const {
    std::size_t seed = std::hash<int>{}(action_id);
    for (int a : arguments) {
      seed ^= std::hash<int>{}(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

class ApplicableActionsGenerator {
public:
  ApplicableActionsGenerator(const Task &task);
  ~ApplicableActionsGenerator();

  std::vector<GroundAction>
  get_applicable_actions(const State &state, bool check_numeric = true) const;

private:
  void set_state(const State &state) const;
  void clear_state() const;

  const Task &_task;
  std::unique_ptr<Clingo::Control> _ctl;
  std::string _asp_program;

  struct PredicateExternalInfo {
    std::string safe_name;
    int arity;
  };
  std::vector<PredicateExternalInfo> _pred_externals;

  bool _has_numeric_preconditions;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEMANTICS_APPLICABLE_ACTIONS_GENERATOR_HH
