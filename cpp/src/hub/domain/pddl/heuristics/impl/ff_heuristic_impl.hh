/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_HEURISTICS_FF_HEURISTIC_IMPL_HH
#define SKDECIDE_PDDL_HEURISTICS_FF_HEURISTIC_IMPL_HH

#include "heuristics/ff_heuristic.hh"

#include "semantics/applicable_actions_generator.hh"
#include "semantics/task.hh"

#include "operator.hh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_set>

namespace skdecide {

namespace pddl {

FFHeuristic::FFHeuristic(const Task &task, double discount_factor,
                         double dead_end_cost, bool verbose)
    : _task(task), _discount_factor(discount_factor),
      _dead_end_cost(dead_end_cost), _verbose(verbose) {
  preground();
}

int FFHeuristic::get_or_create_atom(int predicate_id, const GroundTuple &args) {
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

void FFHeuristic::extract_goal_atoms() {
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

struct FFGroundActionHash {
  std::size_t operator()(const GroundAction &ga) const { return ga.hash(); }
};

void FFHeuristic::preground() {
  ApplicableActionsGenerator aops_gen(_task);
  State relaxed_state = _task.initial_state().copy();

  for (int pid = 0; pid < static_cast<int>(relaxed_state.atoms.size()); ++pid) {
    for (auto &tuple : relaxed_state.atoms[pid]) {
      get_or_create_atom(pid, tuple);
    }
  }

  std::unordered_set<GroundAction, FFGroundActionHash> seen_actions;

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
          {std::move(precond_atoms), std::move(add_atoms), cost, ga});

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
    std::cout << "FFHeuristic: " << _num_atoms << " atoms, "
              << _relaxed_actions.size() << " relaxed actions, "
              << _goal_atoms.size() << " goal atoms" << std::endl;
  }
}

double FFHeuristic::compute(const State &state) const {
  return compute_with_helpful(state).first;
}

std::pair<double, std::vector<GroundAction>>
FFHeuristic::compute_with_helpful(const State &state) const {
  if (_discount_factor >= 1.0) {
    return compute_undiscounted(state);
  }
  return compute_discounted(state);
}

FFHeuristic::ForwardBackwardResult
FFHeuristic::forward_backward_undiscounted(const State &state) const {
  ForwardBackwardResult fbr;
  fbr.g.assign(_num_atoms, std::numeric_limits<double>::infinity());
  fbr.best_supporter.assign(_num_atoms, -1);
  fbr.dead_end = false;

  for (int pid = 0; pid < static_cast<int>(state.atoms.size()); ++pid) {
    for (auto &tuple : state.atoms[pid]) {
      auto it = _atom_index.find({pid, tuple});
      if (it != _atom_index.end()) {
        fbr.g[it->second] = 0.0;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (int ra_idx = 0; ra_idx < static_cast<int>(_relaxed_actions.size());
         ++ra_idx) {
      auto &ra = _relaxed_actions[ra_idx];
      double precond_cost = 0.0;
      bool reachable = true;
      for (int pi : ra.precond_atoms) {
        if (fbr.g[pi] >= _dead_end_cost) {
          reachable = false;
          break;
        }
        precond_cost += fbr.g[pi];
      }
      if (!reachable) {
        continue;
      }
      double action_cost = ra.cost + precond_cost;
      for (int ei : ra.add_atoms) {
        if (action_cost < fbr.g[ei]) {
          fbr.g[ei] = action_cost;
          fbr.best_supporter[ei] = ra_idx;
          changed = true;
        }
      }
    }
  }

  if (_goal_atoms.empty()) {
    return fbr;
  }
  for (int gi : _goal_atoms) {
    if (gi >= _num_atoms || fbr.g[gi] >= _dead_end_cost) {
      fbr.dead_end = true;
      return fbr;
    }
  }

  std::queue<int> open;
  for (int gi : _goal_atoms) {
    if (fbr.g[gi] > 0.0) {
      open.push(gi);
    }
  }

  while (!open.empty()) {
    int p = open.front();
    open.pop();

    if (fbr.marked_atoms.count(p) || fbr.g[p] == 0.0) {
      continue;
    }
    fbr.marked_atoms.insert(p);

    int supporter = fbr.best_supporter[p];
    if (supporter < 0) {
      continue;
    }
    fbr.plan_action_indices.insert(supporter);

    auto &ra = _relaxed_actions[supporter];
    for (int prec : ra.precond_atoms) {
      if (fbr.g[prec] > 0.0 && !fbr.marked_atoms.count(prec)) {
        open.push(prec);
      }
    }
  }

  return fbr;
}

std::pair<double, std::vector<GroundAction>>
FFHeuristic::compute_undiscounted(const State &state) const {
  auto fbr = forward_backward_undiscounted(state);

  if (_goal_atoms.empty()) {
    return {0.0, {}};
  }
  if (fbr.dead_end) {
    return {_dead_end_cost, {}};
  }

  double hff = 0.0;
  for (int idx : fbr.plan_action_indices) {
    hff += _relaxed_actions[idx].cost;
  }

  std::vector<GroundAction> helpful;
  for (int idx : fbr.plan_action_indices) {
    auto &ra = _relaxed_actions[idx];
    bool all_prec_in_state = true;
    for (int prec : ra.precond_atoms) {
      if (fbr.g[prec] != 0.0) {
        all_prec_in_state = false;
        break;
      }
    }
    if (all_prec_in_state) {
      helpful.push_back(ra.ground_action);
    }
  }

  return {hff, std::move(helpful)};
}

FFHeuristicResult FFHeuristic::compute_detailed(const State &state) const {
  auto fbr = forward_backward_undiscounted(state);

  FFHeuristicResult result;

  // Collect reachable atoms with their costs
  for (int i = 0; i < _num_atoms; ++i) {
    if (fbr.g[i] < _dead_end_cost) {
      result.atom_costs.emplace_back(_atoms[i], fbr.g[i]);
    }
  }

  // Goal atom costs
  bool dead_end = fbr.dead_end;
  double goal_cost = 0.0;
  if (!dead_end) {
    for (int gi : _goal_atoms) {
      result.goal_atom_costs.emplace_back(_atoms[gi], fbr.g[gi]);
      goal_cost += fbr.g[gi];
    }
  }

  result.heuristic_value =
      dead_end ? _dead_end_cost : (_goal_atoms.empty() ? 0.0 : 0.0);

  // Relaxed plan actions and helpful actions
  for (int idx : fbr.plan_action_indices) {
    auto &ra = _relaxed_actions[idx];
    result.relaxed_plan_actions.push_back(ra.ground_action);
    bool all_prec_in_state = true;
    for (int prec : ra.precond_atoms) {
      if (fbr.g[prec] != 0.0) {
        all_prec_in_state = false;
        break;
      }
    }
    if (all_prec_in_state) {
      result.helpful_actions.push_back(ra.ground_action);
    }
  }

  // Marked atoms
  for (int ai : fbr.marked_atoms) {
    result.marked_atoms.push_back(_atoms[ai]);
  }

  // Compute hFF = sum of costs of relaxed plan actions
  if (!dead_end && !_goal_atoms.empty()) {
    double hff = 0.0;
    for (int idx : fbr.plan_action_indices) {
      hff += _relaxed_actions[idx].cost;
    }
    result.heuristic_value = hff;
  }

  return result;
}

std::pair<double, std::vector<GroundAction>>
FFHeuristic::compute_discounted(const State &state) const {
  // Forward phase with unit costs for step count
  std::vector<double> g(_num_atoms, std::numeric_limits<double>::infinity());
  std::vector<int> best_supporter(_num_atoms, -1);

  for (int pid = 0; pid < static_cast<int>(state.atoms.size()); ++pid) {
    for (auto &tuple : state.atoms[pid]) {
      auto it = _atom_index.find({pid, tuple});
      if (it != _atom_index.end()) {
        g[it->second] = 0.0;
      }
    }
  }

  // h_add forward chaining with unit costs
  bool changed = true;
  while (changed) {
    changed = false;
    for (int ra_idx = 0; ra_idx < static_cast<int>(_relaxed_actions.size());
         ++ra_idx) {
      auto &ra = _relaxed_actions[ra_idx];
      double precond_steps = 0.0;
      bool reachable = true;
      for (int pi : ra.precond_atoms) {
        if (g[pi] >= _dead_end_cost) {
          reachable = false;
          break;
        }
        precond_steps += g[pi];
      }
      if (!reachable)
        continue;
      double action_steps = 1.0 + precond_steps;
      for (int ei : ra.add_atoms) {
        if (action_steps < g[ei]) {
          g[ei] = action_steps;
          best_supporter[ei] = ra_idx;
          changed = true;
        }
      }
    }
  }

  // Check goal reachability
  if (_goal_atoms.empty())
    return {0.0, {}};

  for (int gi : _goal_atoms) {
    if (gi >= _num_atoms || g[gi] >= _dead_end_cost) {
      double cm = compute_min_cost(state);
      return {cm / (1.0 - _discount_factor), {}};
    }
  }

  // Backward phase: extract relaxed plan
  std::unordered_set<int> marked_atoms;
  std::unordered_set<int> plan_action_indices;
  std::queue<int> open;

  for (int gi : _goal_atoms) {
    if (g[gi] > 0.0) {
      open.push(gi);
    }
  }

  while (!open.empty()) {
    int p = open.front();
    open.pop();

    if (marked_atoms.count(p) || g[p] == 0.0) {
      continue;
    }
    marked_atoms.insert(p);

    int supporter = best_supporter[p];
    if (supporter < 0) {
      continue;
    }
    plan_action_indices.insert(supporter);

    auto &ra = _relaxed_actions[supporter];
    for (int prec : ra.precond_atoms) {
      if (g[prec] > 0.0 && !marked_atoms.count(prec)) {
        open.push(prec);
      }
    }
  }

  // h^{1,+}_FF = number of unique actions in relaxed plan (step count)
  double step_count = static_cast<double>(plan_action_indices.size());

  if (step_count == 0.0)
    return {0.0, {}};

  // Extract helpful actions
  std::vector<GroundAction> helpful;
  for (int idx : plan_action_indices) {
    auto &ra = _relaxed_actions[idx];
    bool all_prec_in_state = true;
    for (int prec : ra.precond_atoms) {
      if (g[prec] != 0.0) {
        all_prec_in_state = false;
        break;
      }
    }
    if (all_prec_in_state) {
      helpful.push_back(ra.ground_action);
    }
  }

  // h^γ_FF(s) = c_m(s) · (1 − γ^{h^{1,+}_FF}) / (1 − γ)
  double cm = compute_min_cost(state);
  double hval = cm * (1.0 - std::pow(_discount_factor, step_count)) /
                (1.0 - _discount_factor);

  return {hval, std::move(helpful)};
}

double FFHeuristic::compute_min_cost(const State &state) const {
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

      double new_cost = std::min(ra.cost, filtered);
      for (int ei : ra.add_atoms) {
        if (new_cost < cm[ei]) {
          cm[ei] = new_cost;
          changed = true;
        }
      }
    }
  }

  double result = std::numeric_limits<double>::infinity();
  for (int i = 0; i < _num_atoms; ++i) {
    if (cm[i] > 0 && cm[i] < result)
      result = cm[i];
  }
  return std::isinf(result) ? _global_min_cost : result;
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_HEURISTICS_FF_HEURISTIC_IMPL_HH
