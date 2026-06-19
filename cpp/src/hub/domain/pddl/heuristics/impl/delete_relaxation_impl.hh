/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_HEURISTICS_DELETE_RELAXATION_IMPL_HH
#define SKDECIDE_PDDL_HEURISTICS_DELETE_RELAXATION_IMPL_HH

#include "heuristics/delete_relaxation.hh"

#include "semantics/applicable_actions_generator.hh"
#include "semantics/task.hh"

#include "operator.hh"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace skdecide {

namespace pddl {

DeleteRelaxationHeuristic::DeleteRelaxationHeuristic(const Task &task,
                                                     HeuristicMode mode,
                                                     double discount_factor,
                                                     double dead_end_cost,
                                                     bool verbose)
    : _task(task), _mode(mode), _discount_factor(discount_factor),
      _dead_end_cost(dead_end_cost), _verbose(verbose) {
  preground();
}

int DeleteRelaxationHeuristic::get_or_create_atom(int predicate_id,
                                                  const GroundTuple &args) {
  FlatAtomKey key{predicate_id, args};
  auto it = _atom_index.find(key);
  if (it != _atom_index.end()) {
    return it->second;
  }
  int idx = _num_atoms++;
  _atom_index[key] = idx;
  _atoms.push_back(key);
  return idx;
}

void DeleteRelaxationHeuristic::extract_goal_atoms() {
  auto &goal = _task.goal();
  if (!goal) {
    return;
  }
  Binding empty_binding;
  goal->collect_positive_atoms(
      _task, empty_binding, [&](int pid, const GroundTuple &gt) {
        _goal_atoms.push_back(get_or_create_atom(pid, gt));
      });
}

struct GroundActionHash {
  std::size_t operator()(const GroundAction &ga) const { return ga.hash(); }
};

void DeleteRelaxationHeuristic::preground() {
  ApplicableActionsGenerator aops_gen(_task);
  State relaxed_state = _task.initial_state().copy();

  // Register initial state atoms in the flat atom table
  for (int pid = 0; pid < static_cast<int>(relaxed_state.atoms.size()); ++pid) {
    for (auto &tuple : relaxed_state.atoms[pid]) {
      get_or_create_atom(pid, tuple);
    }
  }

  std::unordered_set<GroundAction, GroundActionHash> seen_actions;

  bool new_atoms_added = true;
  while (new_atoms_added) {
    new_atoms_added = false;

    auto applicable = aops_gen.get_applicable_actions(relaxed_state, false);

    for (const auto &ga : applicable) {
      if (seen_actions.count(ga)) {
        continue;
      }
      seen_actions.insert(ga);

      auto &action = _task.actions()[ga.action_id];

      std::vector<int> precond_atoms;
      if (action->get_condition()) {
        action->get_condition()->collect_positive_atoms(
            _task, ga.binding, [&](int pid, const GroundTuple &gt) {
              precond_atoms.push_back(get_or_create_atom(pid, gt));
            });
      }

      std::vector<int> add_atoms;
      double cost = 1.0;
      if (action->get_effect()) {
        action->get_effect()->collect_add_atoms(
            _task, ga.binding, [&](int pid, const GroundTuple &gt) {
              add_atoms.push_back(get_or_create_atom(pid, gt));
            });
        action->get_effect()->collect_cost_increase(
            _task, ga.binding, [&](double c) { cost = c; });
      }

      _relaxed_actions.push_back(
          {std::move(precond_atoms), std::move(add_atoms), cost});

      for (int ai : _relaxed_actions.back().add_atoms) {
        auto &atom_key = _atoms[ai];
        auto &atom_set = relaxed_state.atoms[atom_key.predicate_id];
        if (atom_set.insert(atom_key.args).second) {
          new_atoms_added = true;
        }
      }
    }
  }

  extract_goal_atoms();

  _global_min_cost = std::numeric_limits<double>::infinity();
  for (auto &ra : _relaxed_actions) {
    if (ra.cost > 0)
      _global_min_cost = std::min(_global_min_cost, ra.cost);
  }
  if (std::isinf(_global_min_cost))
    _global_min_cost = 1.0;

  if (_verbose) {
    std::cout << "DeleteRelaxationHeuristic: " << _num_atoms << " atoms, "
              << _relaxed_actions.size() << " relaxed actions, "
              << _goal_atoms.size() << " goal atoms" << std::endl;
  }
}

double DeleteRelaxationHeuristic::compute(const State &state) const {
  if (_discount_factor >= 1.0) {
    return compute_undiscounted(state);
  }
  return compute_discounted(state);
}

std::vector<double> DeleteRelaxationHeuristic::forward_chain_undiscounted(
    const State &state) const {
  std::vector<double> g(_num_atoms, std::numeric_limits<double>::infinity());

  for (int pid = 0; pid < static_cast<int>(state.atoms.size()); ++pid) {
    for (auto &tuple : state.atoms[pid]) {
      auto it = _atom_index.find({pid, tuple});
      if (it != _atom_index.end()) {
        g[it->second] = 0.0;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &ra : _relaxed_actions) {
      double precond_cost = 0.0;
      bool reachable = true;
      for (int pi : ra.precond_atoms) {
        if (g[pi] >= _dead_end_cost) {
          reachable = false;
          break;
        }
        if (_mode == HeuristicMode::HADD) {
          precond_cost += g[pi];
        } else {
          precond_cost = std::max(precond_cost, g[pi]);
        }
      }
      if (!reachable) {
        continue;
      }
      double action_cost = ra.cost + precond_cost;
      for (int ei : ra.add_atoms) {
        if (action_cost < g[ei]) {
          g[ei] = action_cost;
          changed = true;
        }
      }
    }
  }

  return g;
}

double
DeleteRelaxationHeuristic::compute_undiscounted(const State &state) const {
  auto g = forward_chain_undiscounted(state);

  if (_goal_atoms.empty()) {
    return 0.0;
  }

  double goal_cost = 0.0;
  for (int gi : _goal_atoms) {
    if (gi >= _num_atoms || g[gi] >= _dead_end_cost) {
      return _dead_end_cost;
    }
    if (_mode == HeuristicMode::HADD) {
      goal_cost += g[gi];
    } else {
      goal_cost = std::max(goal_cost, g[gi]);
    }
  }
  return goal_cost;
}

DeleteRelaxationResult
DeleteRelaxationHeuristic::compute_detailed(const State &state) const {
  auto g = forward_chain_undiscounted(state);

  DeleteRelaxationResult result;

  // Collect reachable atoms with their costs
  for (int i = 0; i < _num_atoms; ++i) {
    if (g[i] < _dead_end_cost) {
      result.atom_costs.emplace_back(_atoms[i], g[i]);
    }
  }

  // Compute goal cost and collect goal atom costs
  double goal_cost = 0.0;
  bool dead_end = false;
  for (int gi : _goal_atoms) {
    if (gi >= _num_atoms || g[gi] >= _dead_end_cost) {
      dead_end = true;
      break;
    }
    result.goal_atom_costs.emplace_back(_atoms[gi], g[gi]);
    if (_mode == HeuristicMode::HADD) {
      goal_cost += g[gi];
    } else {
      goal_cost = std::max(goal_cost, g[gi]);
    }
  }

  result.heuristic_value =
      dead_end ? _dead_end_cost : (_goal_atoms.empty() ? 0.0 : goal_cost);
  return result;
}

double DeleteRelaxationHeuristic::compute_min_cost(const State &state) const {
  // Eq. 8-9: forward chaining tracking minimum non-zero action cost
  std::vector<double> cm(_num_atoms, std::numeric_limits<double>::infinity());

  for (int pid = 0; pid < static_cast<int>(state.atoms.size()); ++pid) {
    for (auto &tuple : state.atoms[pid]) {
      auto it = _atom_index.find({pid, tuple});
      if (it != _atom_index.end()) {
        cm[it->second] = 0.0;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &ra : _relaxed_actions) {
      if (ra.cost <= 0)
        continue;

      // Eq. 9: c̃_s(prec(a)) = cost(a) if prec(a) ⊆ s, else min_{p} c_s(p)
      bool all_in_state = true;
      double prec_min = std::numeric_limits<double>::infinity();
      for (int pi : ra.precond_atoms) {
        if (cm[pi] > 0)
          all_in_state = false;
        prec_min = std::min(prec_min, cm[pi]);
      }
      if (ra.precond_atoms.empty())
        all_in_state = true;

      double filtered = all_in_state ? ra.cost : prec_min;
      if (std::isinf(filtered))
        continue;

      // Eq. 8: c_s(ω) ← min{c_s(ω), cost(a), c̃_s(prec(a))}
      double new_cost = std::min(ra.cost, filtered);
      for (int ei : ra.add_atoms) {
        if (new_cost < cm[ei]) {
          cm[ei] = new_cost;
          changed = true;
        }
      }
    }
  }

  // c_m(s) = min over atoms with c_s > 0
  double result = std::numeric_limits<double>::infinity();
  for (int i = 0; i < _num_atoms; ++i) {
    if (cm[i] > 0 && cm[i] < result)
      result = cm[i];
  }
  return std::isinf(result) ? _global_min_cost : result;
}

double DeleteRelaxationHeuristic::compute_discounted(const State &state) const {
  // Theorem 2 / Definition 4: h^γ_X(s) = c_m(s) · (1 − γ^{h^{1,+}_X(s)}) / (1 −
  // γ)

  // Step 1: forward chain with unit costs → h^{1,+}_X (step count)
  std::vector<double> g(_num_atoms, std::numeric_limits<double>::infinity());

  for (int pid = 0; pid < static_cast<int>(state.atoms.size()); ++pid) {
    for (auto &tuple : state.atoms[pid]) {
      auto it = _atom_index.find({pid, tuple});
      if (it != _atom_index.end()) {
        g[it->second] = 0.0;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &ra : _relaxed_actions) {
      double precond_steps = 0.0;
      bool reachable = true;
      for (int pi : ra.precond_atoms) {
        if (g[pi] >= _dead_end_cost) {
          reachable = false;
          break;
        }
        if (_mode == HeuristicMode::HADD) {
          precond_steps += g[pi];
        } else {
          precond_steps = std::max(precond_steps, g[pi]);
        }
      }
      if (!reachable)
        continue;
      double action_steps = 1.0 + precond_steps;
      for (int ei : ra.add_atoms) {
        if (action_steps < g[ei]) {
          g[ei] = action_steps;
          changed = true;
        }
      }
    }
  }

  // Step 2: aggregate goal step count
  if (_goal_atoms.empty())
    return 0.0;

  double ds = 0.0;
  for (int gi : _goal_atoms) {
    if (gi >= _num_atoms || g[gi] >= _dead_end_cost) {
      // Dead-end: h^γ = c_m / (1 − γ)
      double cm = compute_min_cost(state);
      return cm / (1.0 - _discount_factor);
    }
    if (_mode == HeuristicMode::HADD) {
      ds += g[gi];
    } else {
      ds = std::max(ds, g[gi]);
    }
  }

  if (ds == 0.0)
    return 0.0;

  // Step 3: h^γ_X(s) = c_m(s) · (1 − γ^{h^{1,+}_X}) / (1 − γ)
  double cm = compute_min_cost(state);
  return cm * (1.0 - std::pow(_discount_factor, ds)) / (1.0 - _discount_factor);
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_HEURISTICS_DELETE_RELAXATION_IMPL_HH
