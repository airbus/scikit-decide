/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSPDETHINDSIGHT_IMPL_HH
#define SKDECIDE_SSPDETHINDSIGHT_IMPL_HH

#include "hub/solver/sspdethindsight/sspdethindsight.hh"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "utils/logging.hh"

namespace skdecide {

#define SK_SSPDETHINDSIGHT_TEMPLATE_DECL                                       \
  template <typename Tdomain, typename Texecution_policy,                      \
            typename TdeterminizationAdapter>

#define SK_SSPDETHINDSIGHT_CLASS                                               \
  SSPDetHindsightSolver<Tdomain, Texecution_policy, TdeterminizationAdapter>

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
SK_SSPDETHINDSIGHT_CLASS::SSPDetHindsightSolver(
    Domain &domain, InnerSolverFactory inner_factory,
    AdapterFactory adapter_factory, const GoalCheckerFunctor &goal_checker,
    std::size_t sample_width, double dead_end_cost, std::size_t max_steps,
    double discount, double epsilon, const CallbackFunctor &callback,
    bool verbose)
    : _domain(domain), _inner_factory(std::move(inner_factory)),
      _adapter_factory(std::move(adapter_factory)), _goal_checker(goal_checker),
      _callback(callback), _sample_width(sample_width),
      _dead_end_cost(dead_end_cost), _max_steps(max_steps), _discount(discount),
      _epsilon(epsilon), _verbose(verbose) {}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
void SK_SSPDETHINDSIGHT_CLASS::clear() {
  _policy.clear();
  _has_solution = false;
  _nb_steps = 0;
  _solving_time = 0;
  _explored_states.clear();
  _terminal_states.clear();
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
void SK_SSPDETHINDSIGHT_CLASS::solve(const State &initial_state) {
  _policy.clear();
  _has_solution = false;
  _nb_steps = 0;
  _solving_time = 0;

  _evaluate_hindsight(initial_state);
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
void SK_SSPDETHINDSIGHT_CLASS::_evaluate_hindsight(const State &s) {
  auto start_time = std::chrono::high_resolution_clock::now();

  _explored_states.clear();
  _terminal_states.clear();

  // Collect applicable actions into a vector (Elements may be iterate-only)
  std::vector<Action> orig_action_vec;
  {
    auto orig_actions = _domain.get_applicable_actions(s);
    auto elements = orig_actions.get_elements();
    for (auto a : elements) {
      orig_action_vec.push_back(a);
    }
  }
  std::size_t n_actions = orig_action_vec.size();

  if (n_actions == 0) {
    _has_solution = false;
    if (_verbose) {
      Logger::info("[SSPDetHindsight] Dead end: no applicable actions");
    }
    return;
  }

  _explored_states.insert(s);
  std::vector<double> q_sums(n_actions, 0.0);

  std::vector<std::size_t> scenario_indices(_sample_width);
  std::iota(scenario_indices.begin(), scenario_indices.end(), 0);

  std::for_each(
      Texecution_policy::policy, scenario_indices.begin(),
      scenario_indices.end(), [this, &s, n_actions, &q_sums](std::size_t) {
        Adapter scenario_adapter = _adapter_factory();
        scenario_adapter.update();
        auto &det = scenario_adapter.domain();

        std::vector<DetAction> det_action_vec;
        {
          auto det_actions = det.get_applicable_actions(s);
          auto det_elements = det_actions.get_elements();
          for (auto a : det_elements) {
            det_action_vec.push_back(a);
          }
        }

        std::vector<double> local_q(n_actions, 0.0);
        std::vector<State> local_explored;
        std::vector<State> local_terminal;

        for (std::size_t ai = 0; ai < n_actions; ai++) {
          if (ai >= det_action_vec.size()) {
            local_q[ai] = _dead_end_cost;
            continue;
          }

          auto &da = det_action_vec[ai];
          auto s_prime = det.get_next_state(s, da);
          double cost = det.get_transition_value(s, da, s_prime).cost();
          local_explored.push_back(s_prime);

          if (_goal_checker(_domain, s_prime)) {
            local_q[ai] = cost;
            local_terminal.push_back(s_prime);
          } else {
            auto inner = _inner_factory(det);
            inner->solve(s_prime);
            if (inner->is_solution_defined_for(s_prime)) {
              local_q[ai] = cost + inner->get_best_value(s_prime).cost();
            } else {
              local_q[ai] = _dead_end_cost;
              local_terminal.push_back(s_prime);
            }
          }
        }

        _execution_policy.protect([&]() {
          for (std::size_t ai = 0; ai < n_actions; ai++) {
            q_sums[ai] += local_q[ai];
          }
          for (const auto &e : local_explored) {
            _explored_states.insert(e);
          }
          for (const auto &t : local_terminal) {
            _terminal_states.insert(t);
          }
        });
      });

  std::size_t best_idx = 0;
  double best_q = q_sums[0];
  for (std::size_t ai = 1; ai < n_actions; ai++) {
    if (q_sums[ai] < best_q) {
      best_q = q_sums[ai];
      best_idx = ai;
    }
  }

  _policy[s] = {orig_action_vec[best_idx], best_q / _sample_width};
  _has_solution = true;

  auto end_time = std::chrono::high_resolution_clock::now();
  _solving_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();

  if (_verbose) {
    Logger::info(
        "[SSPDetHindsight] Evaluated " + std::to_string(n_actions) +
        " actions x " + std::to_string(_sample_width) +
        " scenarios, best Q = " + std::to_string(best_q / _sample_width));
  }
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
bool SK_SSPDETHINDSIGHT_CLASS::is_solution_defined_for(const State &s) const {
  return _policy.find(s) != _policy.end();
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
const typename SK_SSPDETHINDSIGHT_CLASS::Action &
SK_SSPDETHINDSIGHT_CLASS::get_best_action(const State &s) {
  if (_nb_steps >= _max_steps) {
    _has_solution = false;
    if (_verbose) {
      Logger::info("[SSPDetHindsight] Max steps (" +
                   std::to_string(_max_steps) + ") reached");
    }
    throw std::runtime_error("SSPDetHindsightSolver: max steps reached");
  }

  _evaluate_hindsight(s);

  if (!_has_solution) {
    throw std::runtime_error(
        "SSPDetHindsightSolver: no solution found (dead end)");
  }

  if (_callback(*this, _domain)) {
    if (_verbose) {
      Logger::info("[SSPDetHindsight] User callback requested stop");
    }
    _has_solution = false;
    throw std::runtime_error("SSPDetHindsightSolver: stopped by callback");
  }

  _nb_steps++;
  return _policy.find(s)->second.first;
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
double SK_SSPDETHINDSIGHT_CLASS::get_best_value(const State &s) const {
  auto it = _policy.find(s);
  if (it == _policy.end()) {
    throw std::runtime_error(
        "SSPDetHindsightSolver: no value defined for this state");
  }
  return it->second.second;
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
std::size_t SK_SSPDETHINDSIGHT_CLASS::get_nb_steps() const { return _nb_steps; }

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
std::size_t SK_SSPDETHINDSIGHT_CLASS::get_solving_time() const {
  return _solving_time;
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
auto SK_SSPDETHINDSIGHT_CLASS::get_explored_states() const -> const StateSet & {
  return _explored_states;
}

SK_SSPDETHINDSIGHT_TEMPLATE_DECL
auto SK_SSPDETHINDSIGHT_CLASS::get_terminal_states() const -> const StateSet & {
  return _terminal_states;
}

} // namespace skdecide

#endif // SKDECIDE_SSPDETHINDSIGHT_IMPL_HH
