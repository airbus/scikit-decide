/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_SSPREPLAN_IMPL_HH
#define SKDECIDE_SSPREPLAN_IMPL_HH

#include "hub/solver/sspreplan/sspreplan.hh"

#include <iostream>
#include <stdexcept>

namespace skdecide {

#define SK_SSPREPLAN_TEMPLATE_DECL                                             \
  template <typename Tdomain, typename Texecution_policy,                      \
            typename TdeterminizationAdapter>

#define SK_SSPREPLAN_CLASS                                                     \
  SSPReplanSolver<Tdomain, Texecution_policy, TdeterminizationAdapter>

SK_SSPREPLAN_TEMPLATE_DECL
SK_SSPREPLAN_CLASS::SSPReplanSolver(
    Domain &domain, Adapter adapter, InnerSolverFactory factory,
    const GoalCheckerFunctor &goal_checker, std::size_t max_replans,
    std::size_t max_steps, const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _adapter(std::move(adapter)),
      _factory(std::move(factory)), _goal_checker(goal_checker),
      _callback(callback), _max_replans(max_replans), _max_steps(max_steps),
      _verbose(verbose) {}

SK_SSPREPLAN_TEMPLATE_DECL
void SK_SSPREPLAN_CLASS::clear() {
  _policy.clear();
  _current_plan.clear();
  _inner_solver.reset();
  _has_solution = false;
  _adapter_initialized = false;
  _nb_replans = 0;
  _nb_steps = 0;
  _solving_time = 0;
  _total_cost = 0.0;
}

SK_SSPREPLAN_TEMPLATE_DECL
void SK_SSPREPLAN_CLASS::solve(const State &initial_state) {
  _policy.clear();
  _current_plan.clear();
  _inner_solver.reset();
  _has_solution = false;
  _nb_replans = 0;
  _nb_steps = 0;
  _total_cost = 0.0;
  _solving_time = 0;
  _adapter_initialized = false;

  _replan(initial_state);
}

SK_SSPREPLAN_TEMPLATE_DECL
void SK_SSPREPLAN_CLASS::_replan(const State &s) {
  if (_nb_replans >= _max_replans) {
    _has_solution = false;
    if (_verbose) {
      std::cout << "[SSPReplan] Max replans (" << _max_replans << ") reached"
                << std::endl;
    }
    return;
  }

  if (!_adapter_initialized || _adapter.needs_update_each_replan()) {
    _adapter.update();
    _adapter_initialized = true;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  _inner_solver = _factory(_adapter.domain());
  _inner_solver->solve(s);
  _nb_replans++;

  auto end_time = std::chrono::high_resolution_clock::now();
  _solving_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();

  _has_solution = _inner_solver->is_solution_defined_for(s);
  _expected_current = s;

  _current_plan.clear();
  if (_has_solution) {
    State current = s;
    std::size_t safety = 0;
    while (_inner_solver->is_solution_defined_for(current) &&
           !_goal_checker(_domain, current) && safety++ < _max_steps) {
      const DetAction &det_a = _inner_solver->get_best_action(current);
      Action orig_a = _adapter.to_original(det_a);
      _current_plan.push_back({current, orig_a});
      current = _adapter.expected_next(current, det_a);
    }
  }

  if (_verbose) {
    std::cout << "[SSPReplan] Replan #" << _nb_replans << " from state"
              << std::endl;
  }

  if (_has_solution && _callback(*this, _domain)) {
    if (_verbose) {
      std::cout << "[SSPReplan] User callback requested stop" << std::endl;
    }
    _has_solution = false;
  }
}

SK_SSPREPLAN_TEMPLATE_DECL
bool SK_SSPREPLAN_CLASS::is_solution_defined_for(const State &s) const {
  return _policy.find(s) != _policy.end();
}

SK_SSPREPLAN_TEMPLATE_DECL
const typename SK_SSPREPLAN_CLASS::Action &
SK_SSPREPLAN_CLASS::get_best_action(const State &s) {
  bool need_replan = false;

  if (!_has_solution || !_inner_solver) {
    need_replan = true;
  } else if (!_inner_solver->is_solution_defined_for(s)) {
    need_replan = true;
  } else if (!typename State::Equal()(s, _expected_current)) {
    need_replan = true;
    if (_verbose) {
      std::cout << "[SSPReplan] Deviation at step " << _nb_steps
                << ", replanning" << std::endl;
    }
  }

  if (need_replan) {
    _replan(s);
  }

  if (!_has_solution || !_inner_solver ||
      !_inner_solver->is_solution_defined_for(s)) {
    throw std::runtime_error(
        "SSPReplanSolver: no solution found (dead end or limits reached)");
  }

  const DetAction &det_action = _inner_solver->get_best_action(s);
  Action orig_action = _adapter.to_original(det_action);
  _policy[s] = {orig_action, 0.0};

  _expected_current = _adapter.expected_next(s, det_action);
  _nb_steps++;

  if (_nb_steps >= _max_steps) {
    _has_solution = false;
    if (_verbose) {
      std::cout << "[SSPReplan] Max steps (" << _max_steps << ") reached"
                << std::endl;
    }
  }

  return _policy.find(s)->second.first;
}

SK_SSPREPLAN_TEMPLATE_DECL
double SK_SSPREPLAN_CLASS::get_best_value(const State &s) const {
  auto it = _policy.find(s);
  if (it == _policy.end()) {
    throw std::runtime_error(
        "SSPReplanSolver: no value defined for this state");
  }
  return it->second.second;
}

SK_SSPREPLAN_TEMPLATE_DECL
const typename SK_SSPREPLAN_CLASS::Plan &SK_SSPREPLAN_CLASS::get_plan() const {
  return _current_plan;
}

SK_SSPREPLAN_TEMPLATE_DECL
std::size_t SK_SSPREPLAN_CLASS::get_nb_replans() const { return _nb_replans; }

SK_SSPREPLAN_TEMPLATE_DECL
std::size_t SK_SSPREPLAN_CLASS::get_nb_steps() const { return _nb_steps; }

SK_SSPREPLAN_TEMPLATE_DECL
std::size_t SK_SSPREPLAN_CLASS::get_solving_time() const {
  return _solving_time;
}

SK_SSPREPLAN_TEMPLATE_DECL
double SK_SSPREPLAN_CLASS::get_total_cost() const { return _total_cost; }

} // namespace skdecide

#endif // SKDECIDE_SSPREPLAN_IMPL_HH
