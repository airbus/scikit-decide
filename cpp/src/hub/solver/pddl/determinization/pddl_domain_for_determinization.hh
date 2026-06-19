/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DOMAIN_FOR_DETERMINIZATION_HH
#define SKDECIDE_PDDL_DOMAIN_FOR_DETERMINIZATION_HH

#include <vector>

#include "hub/domain/pddl/semantics/applicable_actions_generator.hh"
#include "hub/domain/pddl/semantics/successor_generator.hh"
#include "hub/domain/pddl/semantics/task.hh"
#include "hub/solver/pddl/pddl_domain_adapter.hh"

namespace skdecide {

namespace pddl {

/**
 * @brief Stochastic domain adapter for PPDDL tasks.
 *
 * Wraps a parsed PPDDL Task to provide the stochastic domain interface
 * (applicable actions, next-state distribution, transition cost) expected
 * by meta-solvers (SSPReplan, SSPPlanMerger, SSPDetHindsight).
 *
 * Unlike PddlDeterministicDomain, this adapter exposes probabilistic
 * successor distributions via get_next_state_distribution(), reflecting
 * the stochastic effects in the PPDDL task.
 */
class PddlDomainForDeterminization {
public:
  using State = PddlState;
  using Action = PddlAction;
  using Value = PddlValue;
  using Predicate = bool;

  struct ActionSpace {
    std::vector<Action> _actions;
    const std::vector<Action> &get_elements() const { return _actions; }
  };

  struct DistributionValue {
    PddlState _state;
    double _probability;
    const PddlState &state() const { return _state; }
    const double &probability() const { return _probability; }
  };

  struct NextStateDistributionValues {
    std::vector<DistributionValue> _values;
    auto begin() const { return _values.begin(); }
    auto end() const { return _values.end(); }
  };

  struct NextStateDistribution {
    std::vector<DistributionValue> _values;
    NextStateDistributionValues get_values() const { return {_values}; }
  };

  /**
   * @param task Parsed PPDDL task providing action schemas with stochastic
   *        effects, initial state, goal, and (optional) total-cost function.
   */
  PddlDomainForDeterminization(const Task &task);

  ActionSpace get_applicable_actions(const State &s) const;
  NextStateDistribution get_next_state_distribution(const State &s,
                                                    const Action &a) const;
  Value get_transition_value(const State &s, const Action &a,
                             const State &ns) const;

  const Task &task() const { return _task; }

private:
  const Task &_task;
  mutable ApplicableActionsGenerator _aops_gen;
  mutable SuccessorGenerator _succ_gen;
  int _total_cost_idx = -1;
};

} // namespace pddl

} // namespace skdecide

#include "impl/pddl_domain_for_determinization_impl.hh"

#endif // SKDECIDE_PDDL_DOMAIN_FOR_DETERMINIZATION_HH
