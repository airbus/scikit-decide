/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_HEURISTICS_FF_HEURISTIC_HH
#define SKDECIDE_PDDL_HEURISTICS_FF_HEURISTIC_HH

#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "heuristics/delete_relaxation.hh"
#include "semantics/applicable_actions_generator.hh"
#include "semantics/state.hh"

namespace skdecide {

namespace pddl {

class Task;

struct FFRelaxedAction {
  std::vector<int> precond_atoms;
  std::vector<int> add_atoms;
  double cost;
  GroundAction ground_action;
};

struct FFHeuristicResult {
  double heuristic_value;
  std::vector<std::pair<FlatAtomKey, double>> atom_costs;
  std::vector<std::pair<FlatAtomKey, double>> goal_atom_costs;
  std::vector<GroundAction> relaxed_plan_actions;
  std::vector<GroundAction> helpful_actions;
  std::vector<FlatAtomKey> marked_atoms;
};

/**
 * h_FF delete-relaxation heuristic.
 *
 * Builds a relaxed planning graph via h_add forward chaining, then extracts
 * a relaxed plan backwards from the goal as described in:
 *
 *   Hoffmann, J. and Nebel, B. (2001). The FF Planning System:
 *   Fast Plan Generation Through Heuristic Search.
 *   Journal of Artificial Intelligence Research, 14, 253-302.
 */
class FFHeuristic {
public:
  FFHeuristic(const Task &task, double discount_factor = 1.0,
              double dead_end_cost = 1e9, bool verbose = false);

  double compute(const State &state) const;
  std::pair<double, std::vector<GroundAction>>
  compute_with_helpful(const State &state) const;
  FFHeuristicResult compute_detailed(const State &state) const;

  double discount_factor() const { return _discount_factor; }
  int num_atoms() const { return _num_atoms; }
  int num_relaxed_actions() const {
    return static_cast<int>(_relaxed_actions.size());
  }

private:
  const Task &_task;
  double _discount_factor;
  double _dead_end_cost;
  bool _verbose;
  double _global_min_cost = 1.0;

  std::unordered_map<FlatAtomKey, int, FlatAtomKeyHash> _atom_index;
  std::vector<FlatAtomKey> _atoms;
  int _num_atoms = 0;

  std::vector<FFRelaxedAction> _relaxed_actions;
  std::vector<int> _goal_atoms;

  void preground();
  int get_or_create_atom(int predicate_id, const GroundTuple &args);

  void extract_goal_atoms();

  struct ForwardBackwardResult {
    std::vector<double> g;
    std::vector<int> best_supporter;
    std::unordered_set<int> marked_atoms;
    std::unordered_set<int> plan_action_indices;
    bool dead_end;
  };

  ForwardBackwardResult forward_backward_undiscounted(const State &state) const;

  std::pair<double, std::vector<GroundAction>>
  compute_undiscounted(const State &state) const;
  std::pair<double, std::vector<GroundAction>>
  compute_discounted(const State &state) const;
  double compute_min_cost(const State &state) const;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_HEURISTICS_FF_HEURISTIC_HH
