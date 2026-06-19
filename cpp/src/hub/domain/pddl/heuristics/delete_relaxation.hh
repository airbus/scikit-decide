/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_HEURISTICS_DELETE_RELAXATION_HH
#define SKDECIDE_PDDL_HEURISTICS_DELETE_RELAXATION_HH

#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "semantics/state.hh"

namespace skdecide {

namespace pddl {

class Task;

enum class HeuristicMode { HADD, HMAX };

struct FlatAtomKey {
  int predicate_id;
  GroundTuple args;

  bool operator==(const FlatAtomKey &other) const {
    return predicate_id == other.predicate_id && args == other.args;
  }
};

struct FlatAtomKeyHash {
  std::size_t operator()(const FlatAtomKey &k) const {
    std::size_t seed = std::hash<int>{}(k.predicate_id);
    for (int v : k.args) {
      seed ^= std::hash<int>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

struct RelaxedAction {
  std::vector<int> precond_atoms;
  std::vector<int> add_atoms;
  double cost;
};

struct DeleteRelaxationResult {
  double heuristic_value;
  std::vector<std::pair<FlatAtomKey, double>> atom_costs;
  std::vector<std::pair<FlatAtomKey, double>> goal_atom_costs;
};

class DeleteRelaxationHeuristic {
public:
  DeleteRelaxationHeuristic(const Task &task, HeuristicMode mode,
                            double discount_factor = 1.0,
                            double dead_end_cost = 1e9, bool verbose = false);

  double compute(const State &state) const;
  DeleteRelaxationResult compute_detailed(const State &state) const;

  HeuristicMode mode() const { return _mode; }
  double discount_factor() const { return _discount_factor; }
  int num_atoms() const { return _num_atoms; }
  int num_relaxed_actions() const {
    return static_cast<int>(_relaxed_actions.size());
  }

private:
  const Task &_task;
  HeuristicMode _mode;
  double _discount_factor;
  double _dead_end_cost;
  bool _verbose;
  double _global_min_cost = 1.0;

  std::unordered_map<FlatAtomKey, int, FlatAtomKeyHash> _atom_index;
  std::vector<FlatAtomKey> _atoms;
  int _num_atoms = 0;

  std::vector<RelaxedAction> _relaxed_actions;
  std::vector<int> _goal_atoms;

  void preground();
  int get_or_create_atom(int predicate_id, const GroundTuple &args);

  void extract_goal_atoms();

  std::vector<double> forward_chain_undiscounted(const State &state) const;
  double compute_undiscounted(const State &state) const;
  double compute_discounted(const State &state) const;
  double compute_min_cost(const State &state) const;
};

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_HEURISTICS_DELETE_RELAXATION_HH
