/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DOMAIN_ADAPTER_HH
#define SKDECIDE_PDDL_DOMAIN_ADAPTER_HH

#include <string>
#include <vector>

#include "hub/domain/pddl/semantics/applicable_actions_generator.hh"
#include "hub/domain/pddl/semantics/state.hh"
#include "hub/domain/pddl/semantics/successor_generator.hh"
#include "hub/domain/pddl/semantics/task.hh"

namespace skdecide {

namespace pddl {

struct PddlState : State {
  using State::State;
  PddlState() = default;
  PddlState(const State &s) : State(s) {}
  PddlState(State &&s) : State(std::move(s)) {}

  struct Hash {
    std::size_t operator()(const PddlState &s) const { return s.hash(); }
  };

  struct Equal {
    bool operator()(const PddlState &a, const PddlState &b) const {
      return a == b;
    }
  };

  std::string print() const {
    return "PddlState(hash=" + std::to_string(hash()) + ")";
  }
};

struct PddlAction : GroundAction {
  using GroundAction::GroundAction;
  PddlAction() = default;
  PddlAction(const GroundAction &a) : GroundAction(a) {}
  PddlAction(GroundAction &&a) : GroundAction(std::move(a)) {}

  struct Hash {
    std::size_t operator()(const PddlAction &a) const { return a.hash(); }
  };

  struct Equal {
    bool operator()(const PddlAction &a, const PddlAction &b) const {
      return a == b;
    }
  };

  std::string print() const {
    return "PddlAction(id=" + std::to_string(action_id) + ")";
  }
};

struct PddlValue {
  double _cost = 0.0;

  PddlValue() = default;
  explicit PddlValue(double c) : _cost(c) {}

  double cost() const { return _cost; }
  void cost(double c) { _cost = c; }

  std::string print() const { return std::to_string(_cost); }
};

/**
 * @brief Deterministic domain adapter for PDDL tasks.
 *
 * Wraps a parsed PDDL Task to provide the deterministic domain interface
 * (applicable actions, successor state, transition cost) expected by
 * scikit-decide C++ solvers such as EHC/FF and the inner solver registry.
 *
 * Transition cost is derived from the PDDL total-cost function when present;
 * otherwise a unit cost of 1.0 is used.
 */
class PddlDeterministicDomain {
public:
  using State = PddlState;
  using Action = PddlAction;
  using Value = PddlValue;
  using Predicate = bool;

  struct ActionSpace {
    std::vector<Action> _actions;
    const std::vector<Action> &get_elements() const { return _actions; }
  };

  /**
   * @param task Parsed PDDL task providing the action schemas, initial state,
   *        goal description, and (optional) total-cost numeric function.
   */
  PddlDeterministicDomain(const Task &task);

  ActionSpace get_applicable_actions(const State &s) const;
  State get_next_state(const State &s, const Action &a) const;
  const Task &task() const { return _task; }
  Value get_transition_value(const State &s, const Action &a,
                             const State &ns) const;

private:
  const Task &_task;
  mutable ApplicableActionsGenerator _aops_gen;
  SuccessorGenerator _succ_gen;
  int _total_cost_idx = -1;
};

} // namespace pddl

} // namespace skdecide

#include "impl/pddl_domain_adapter_impl.hh"

#endif // SKDECIDE_PDDL_DOMAIN_ADAPTER_HH
