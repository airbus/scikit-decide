/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSPPLANMERGER_IMPL_HH
#define SKDECIDE_SSPPLANMERGER_IMPL_HH

#include "hub/solver/sspplanmerger/sspplanmerger.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>

#include "utils/logging.hh"

namespace skdecide {

#define SK_SSPPLANMERGER_TEMPLATE_DECL                                         \
  template <typename Tdomain, typename Texecution_policy,                      \
            typename TdeterminizationAdapter>

#define SK_SSPPLANMERGER_CLASS                                                 \
  SSPPlanMergerSolver<Tdomain, Texecution_policy, TdeterminizationAdapter>

SK_SSPPLANMERGER_TEMPLATE_DECL
SK_SSPPLANMERGER_CLASS::SSPPlanMergerSolver(
    Domain &domain, AdapterFactory adapter_factory,
    InnerSolverFactory inner_factory, const GoalCheckerFunctor &goal_checker,
    double rho, std::size_t mc_samples, std::size_t max_iterations,
    std::size_t max_steps, double dead_end_cost, bool optimize_policy_graph,
    double discount, double epsilon, const CallbackFunctor &callback,
    bool verbose)
    : _domain(domain), _adapter_factory(std::move(adapter_factory)),
      _inner_factory(std::move(inner_factory)), _goal_checker(goal_checker),
      _callback(callback), _rho(rho), _mc_samples(mc_samples),
      _max_iterations(max_iterations), _max_steps(max_steps),
      _dead_end_cost(dead_end_cost),
      _optimize_policy_graph(optimize_policy_graph), _discount(discount),
      _epsilon(epsilon), _verbose(verbose) {}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::clear() {
  _policy.clear();
  _values_evaluated = false;
  _has_solution = false;
  _nb_iterations = 0;
  _nb_plans = 0;
  _solving_time = 0;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::solve(const State &s0) {
  clear();
  resolve(s0);
}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::resolve(const State &s0) {
  _values_evaluated = false;
  auto start_time = std::chrono::high_resolution_clock::now();

  _plan_from(s0);

  for (std::size_t iter = 0; iter < _max_iterations; iter++) {
    _nb_iterations++;

    if (_optimize_policy_graph) {
      _optimize_ssp(s0);
    }

    std::vector<State> terminals;
    std::size_t n_replan = 0;

    for (std::size_t i = 0; i < _mc_samples; i++) {
      State s = s0;
      std::size_t steps = 0;
      while (!_goal_checker(_domain, s) && steps < _max_steps) {
        auto it = _policy.find(s);
        if (it == _policy.end()) {
          n_replan++;
          terminals.push_back(s);
          break;
        }
        s = _sample_successor(s, it->second.first);
        steps++;
      }
    }

    double p_replan = static_cast<double>(n_replan) / _mc_samples;

    if (_verbose) {
      Logger::info("[SSPPlanMerger] Iter " + std::to_string(_nb_iterations) +
                   ": p_replan=" + std::to_string(p_replan) +
                   ", policy_size=" + std::to_string(_policy.size()));
    }

    if (p_replan <= _rho) {
      break;
    }

    if (_callback(*this, _domain)) {
      if (_verbose) {
        Logger::info("[SSPPlanMerger] User callback requested stop");
      }
      break;
    }

    if (terminals.empty()) {
      continue;
    }

    _plan_from_terminals(terminals);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  _solving_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();

  _has_solution = !_policy.empty();
}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::_plan_from(const State &s) {
  Adapter adapter = _adapter_factory();
  adapter.update();
  auto inner = _inner_factory(adapter.domain());
  inner->solve(s);
  _nb_plans++;

  State current = s;
  while (inner->is_solution_defined_for(current)) {
    const DetAction &det_a = inner->get_best_action(current);
    Action orig_a = adapter.to_original(det_a);
    double val = inner->get_best_value(current).cost();
    _policy[current] = {orig_a, val};
    current = adapter.expected_next(current, det_a);
    if (_goal_checker(_domain, current)) {
      break;
    }
  }
}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::_plan_from_terminals(
    const std::vector<State> &terminals) {
  using StateSet = typename SetTypeDeducer<State>::Set;
  StateSet seen;
  for (const auto &t : terminals) {
    if (seen.find(t) != seen.end()) {
      continue;
    }
    seen.insert(t);
    _plan_from(t);
  }
}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::_optimize_ssp(const State &s0) {
  using StateVec = std::vector<State>;
  using StateSet = typename SetTypeDeducer<State>::Set;
  using ValueMap = typename MapTypeDeducer<State, double>::Map;

  StateVec reachable;
  StateSet visited;

  std::queue<State> bfs_queue;
  if (_policy.find(s0) != _policy.end()) {
    bfs_queue.push(s0);
    visited.insert(s0);
  }

  while (!bfs_queue.empty()) {
    State s = bfs_queue.front();
    bfs_queue.pop();
    reachable.push_back(s);

    auto pit = _policy.find(s);
    if (pit == _policy.end()) {
      continue;
    }

    const Action &a = pit->second.first;
    auto dist = _domain.get_next_state_distribution(s, a);
    auto values = dist.get_values();
    for (const auto &dv : values) {
      const State &sp = dv.state();
      if (visited.find(sp) == visited.end() &&
          _policy.find(sp) != _policy.end() && !_goal_checker(_domain, sp)) {
        visited.insert(sp);
        bfs_queue.push(sp);
      }
    }
  }

  if (reachable.empty()) {
    return;
  }

  ValueMap V;
  for (const auto &s : reachable) {
    auto pit = _policy.find(s);
    V[s] = pit->second.second;
  }

  double max_delta;
  do {
    max_delta = 0.0;

    for (const auto &s : reachable) {
      auto aops = _domain.get_applicable_actions(s);
      auto elements = aops.get_elements();

      double best_q = std::numeric_limits<double>::infinity();
      Action best_action = _policy.find(s)->second.first;

      for (const auto &a : elements) {
        auto dist = _domain.get_next_state_distribution(s, a);
        auto dvalues = dist.get_values();

        double expected = 0.0;
        bool has_successor = false;
        State first_sp = s;

        for (const auto &dv : dvalues) {
          has_successor = true;
          first_sp = dv.state();
          const State &sp = dv.state();
          double prob = dv.probability();

          double v_sp;
          if (_goal_checker(_domain, sp)) {
            v_sp = 0.0;
          } else {
            auto vit = V.find(sp);
            if (vit != V.end()) {
              v_sp = vit->second;
            } else {
              v_sp = _dead_end_cost;
            }
          }
          expected += prob * v_sp;
        }

        if (!has_successor) {
          continue;
        }

        double cost = _domain.get_transition_value(s, a, first_sp).cost();
        double q_val = cost + _discount * expected;

        if (q_val < best_q) {
          best_q = q_val;
          best_action = a;
        }
      }

      if (best_q < std::numeric_limits<double>::infinity()) {
        double old_v = V[s];
        V[s] = best_q;
        _policy[s] = {best_action, best_q};
        max_delta = std::max(max_delta, std::abs(best_q - old_v));
      }
    }
  } while (max_delta >= _epsilon);

  StateVec to_remove;
  for (auto &[s, av] : _policy) {
    if (av.second >= _dead_end_cost) {
      to_remove.push_back(s);
    }
  }
  for (const auto &s : to_remove) {
    _policy.erase(s);
  }
}

SK_SSPPLANMERGER_TEMPLATE_DECL
typename SK_SSPPLANMERGER_CLASS::State
SK_SSPPLANMERGER_CLASS::_sample_successor(const State &s, const Action &a) {
  auto dist = _domain.get_next_state_distribution(s, a);
  auto values = dist.get_values();

  std::vector<double> weights;
  std::vector<State> states;
  for (const auto &dv : values) {
    states.push_back(dv.state());
    weights.push_back(dv.probability());
  }

  if (states.empty()) {
    return s;
  }

  std::discrete_distribution<> d(weights.begin(), weights.end());
  return states[d(_rng)];
}

SK_SSPPLANMERGER_TEMPLATE_DECL
bool SK_SSPPLANMERGER_CLASS::is_solution_defined_for(const State &s) const {
  return _policy.find(s) != _policy.end();
}

SK_SSPPLANMERGER_TEMPLATE_DECL
const typename SK_SSPPLANMERGER_CLASS::Action &
SK_SSPPLANMERGER_CLASS::get_best_action(const State &s) {
  auto it = _policy.find(s);
  if (it == _policy.end()) {
    if (_verbose) {
      Logger::info("[SSPPlanMerger] State not in policy, resolving");
    }
    resolve(s);
    it = _policy.find(s);
    if (it == _policy.end()) {
      throw std::runtime_error(
          "SSPPlanMergerSolver: state not in policy even after resolving");
    }
  }
  return it->second.first;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
void SK_SSPPLANMERGER_CLASS::_evaluate_policy() const {
  using ValueMap = typename MapTypeDeducer<State, double>::Map;

  std::vector<State> states;
  states.reserve(_policy.size());
  for (const auto &entry : _policy) {
    states.push_back(entry.first);
  }

  if (states.empty()) {
    _values_evaluated = true;
    return;
  }

  ValueMap V;
  for (const auto &s : states) {
    V[s] = _policy.find(s)->second.second;
  }

  double max_delta;
  do {
    max_delta = 0.0;

    for (const auto &s : states) {
      auto pit = _policy.find(s);
      const Action &a = pit->second.first;

      auto dist = _domain.get_next_state_distribution(s, a);
      auto dvalues = dist.get_values();

      double expected = 0.0;
      bool has_successor = false;
      State first_sp = s;

      for (const auto &dv : dvalues) {
        has_successor = true;
        first_sp = dv.state();
        const State &sp = dv.state();
        double prob = dv.probability();

        double v_sp;
        if (_goal_checker(_domain, sp)) {
          v_sp = 0.0;
        } else {
          auto vit = V.find(sp);
          if (vit != V.end()) {
            v_sp = vit->second;
          } else {
            v_sp = _dead_end_cost;
          }
        }
        expected += prob * v_sp;
      }

      if (!has_successor) {
        continue;
      }

      double cost = _domain.get_transition_value(s, a, first_sp).cost();
      double new_v = cost + _discount * expected;

      double old_v = V[s];
      V[s] = new_v;
      max_delta = std::max(max_delta, std::abs(new_v - old_v));
    }
  } while (max_delta >= _epsilon);

  for (const auto &s : states) {
    _policy.find(s)->second.second = V[s];
  }

  _values_evaluated = true;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
double SK_SSPPLANMERGER_CLASS::get_best_value(const State &s) const {
  if (!_optimize_policy_graph && !_values_evaluated) {
    _evaluate_policy();
  }
  auto it = _policy.find(s);
  if (it == _policy.end()) {
    throw std::runtime_error(
        "SSPPlanMergerSolver: no value defined for this state");
  }
  return it->second.second;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
std::size_t SK_SSPPLANMERGER_CLASS::get_nb_iterations() const {
  return _nb_iterations;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
std::size_t SK_SSPPLANMERGER_CLASS::get_nb_plans() const { return _nb_plans; }

SK_SSPPLANMERGER_TEMPLATE_DECL
std::size_t SK_SSPPLANMERGER_CLASS::get_solving_time() const {
  return _solving_time;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
std::size_t SK_SSPPLANMERGER_CLASS::get_policy_size() const {
  return _policy.size();
}

SK_SSPPLANMERGER_TEMPLATE_DECL
auto SK_SSPPLANMERGER_CLASS::get_explored_states() const ->
    typename SetTypeDeducer<State>::Set {
  typename SetTypeDeducer<State>::Set states;
  for (const auto &entry : _policy) {
    states.insert(entry.first);
  }
  return states;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
auto SK_SSPPLANMERGER_CLASS::get_terminal_states() const ->
    typename SetTypeDeducer<State>::Set {
  typename SetTypeDeducer<State>::Set terminals;
  for (const auto &entry : _policy) {
    const State &s = entry.first;
    const Action &a = entry.second.first;
    auto dist = _domain.get_next_state_distribution(s, a);
    auto values = dist.get_values();
    for (const auto &dv : values) {
      const State &sp = dv.state();
      if (_policy.find(sp) == _policy.end() && !_goal_checker(_domain, sp)) {
        terminals.insert(sp);
      }
    }
  }
  return terminals;
}

SK_SSPPLANMERGER_TEMPLATE_DECL
auto SK_SSPPLANMERGER_CLASS::get_policy() const -> PolicyMap { return _policy; }

} // namespace skdecide

#endif // SKDECIDE_SSPPLANMERGER_IMPL_HH
