/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of SSiPP (Algorithm 2) from:
 * Trevizan & Veloso, "Short-Sighted Stochastic Shortest Path Problems",
 * ICAPS 2012.
 */
#ifndef SKDECIDE_SSIPP_IMPL_HH
#define SKDECIDE_SSIPP_IMPL_HH

#include <algorithm>
#include <limits>
#include <type_traits>

#include "hub/solver/ilaostar/ilaostar.hh"
#include "hub/solver/ilaostar/impl/ilaostar_impl.hh"
#include "hub/solver/ldfs/ldfs.hh"
#include "hub/solver/ldfs/impl/ldfs_impl.hh"
#include "hub/solver/lrtdp/lrtdp.hh"
#include "hub/solver/lrtdp/impl/lrtdp_impl.hh"
#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_SSIPP_TEMPLATE_DECL                                                 \
  template <typename Tdomain, typename Texecution_policy,                      \
            template <typename, typename> class TinnerSolver>

#define SK_SSIPP_CLASS SSiPPSolver<Tdomain, Texecution_policy, TinnerSolver>

SK_SSIPP_TEMPLATE_DECL
template <typename... InnerSolverArgs>
SK_SSIPP_CLASS::SSiPPSolver(Domain &domain,
                            const GoalCheckerFunctor &goal_checker,
                            const HeuristicFunctor &heuristic,
                            std::size_t depth, double discount, double epsilon,
                            std::size_t max_iterations,
                            const CallbackFunctor &callback, bool verbose,
                            InnerSolverArgs &&...inner_solver_args)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _depth(depth), _discount(discount), _epsilon(epsilon),
      _max_iterations(max_iterations), _callback(callback), _verbose(verbose),
      _nb_sub_ssps(0) {

  auto captured_args =
      std::make_tuple(std::forward<InnerSolverArgs>(inner_solver_args)...);

  if constexpr (std::is_same_v<InnerSolver,
                               LRTDPSolver<Tdomain, Texecution_policy>>) {
    _inner_solver_factory = [captured_args = std::move(captured_args)](
                                Domain &d, GoalCheckerFunctor sub_gc,
                                HeuristicFunctor sub_h) mutable {
      auto wrapped_gc = [sub_gc](Domain &dd, const State &st,
                                 const std::size_t *) -> Predicate {
        return sub_gc(dd, st);
      };
      auto wrapped_h = [sub_h](Domain &dd, const State &st,
                               const std::size_t *) -> Value {
        return sub_h(dd, st);
      };
      return std::apply(
          [&](auto &&...args) {
            return std::make_unique<InnerSolver>(
                d, wrapped_gc, wrapped_h,
                std::forward<decltype(args)>(args)...);
          },
          captured_args);
    };
  } else {
    _inner_solver_factory = [captured_args = std::move(captured_args)](
                                Domain &d, GoalCheckerFunctor sub_gc,
                                HeuristicFunctor sub_h) mutable {
      return std::apply(
          [&](auto &&...args) {
            return std::make_unique<InnerSolver>(
                d, sub_gc, sub_h, std::forward<decltype(args)>(args)...);
          },
          captured_args);
    };
  }

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm SSiPP");
  }
}

SK_SSIPP_TEMPLATE_DECL
void SK_SSIPP_CLASS::clear() {
  _value_function.clear();
  _policy.clear();
  _current_subssp_states.clear();
  _boundary_states.clear();
  _nb_sub_ssps = 0;
}

SK_SSIPP_TEMPLATE_DECL
double SK_SSIPP_CLASS::get_value(const State &s) const {
  auto it = _value_function.find(s);
  if (it != _value_function.end()) {
    return it->second;
  }
  return _heuristic(_domain, s).cost();
}

// --- BFS to build short-sighted sub-SSP ---

SK_SSIPP_TEMPLATE_DECL
void SK_SSIPP_CLASS::build_short_sighted_ssp(const State &s) {
  _current_subssp_states.clear();
  _boundary_states.clear();

  typename MapTypeDeducer<State, std::size_t>::Map distance;
  std::queue<State> queue;

  distance[s] = 0;
  queue.push(s);

  while (!queue.empty()) {
    State current = queue.front();
    queue.pop();
    std::size_t d = distance[current];

    _current_subssp_states.insert(current);

    if (_goal_checker(_domain, current)) {
      _boundary_states.insert(current);
      continue;
    }

    if (d >= _depth) {
      _boundary_states.insert(current);
      continue;
    }

    auto actions = _domain.get_applicable_actions(current).get_elements();
    for (auto a : actions) {
      auto next_dist =
          _domain.get_next_state_distribution(current, a).get_values();
      for (auto ns : next_dist) {
        if (distance.find(ns.state()) == distance.end()) {
          distance[ns.state()] = d + 1;
          queue.push(ns.state());
        }
      }
    }
  }

  if (_verbose) {
    Logger::debug(
        "SSiPP: built sub-SSP with " +
        StringConverter::from(_current_subssp_states.size()) + " states and " +
        StringConverter::from(_boundary_states.size()) + " boundary states");
  }
}

// --- Solve sub-SSP with inner solver ---

SK_SSIPP_TEMPLATE_DECL
void SK_SSIPP_CLASS::solve_subssp(const State &s) {
  auto sub_goal_checker = [this](Domain &d, const State &st) -> Predicate {
    return _boundary_states.find(st) != _boundary_states.end();
  };

  auto sub_heuristic = [this](Domain &d, const State &st) -> Value {
    Value v;
    v.cost(get_value(st));
    return v;
  };

  auto inner = _inner_solver_factory(_domain, sub_goal_checker, sub_heuristic);
  inner->solve(s);

  // Extract policy from inner solver
  for (const auto &st : _current_subssp_states) {
    if (inner->is_solution_defined_for(st) &&
        _boundary_states.find(st) == _boundary_states.end()) {
      try {
        _policy[st] = inner->get_best_action(st);
      } catch (...) {
      }
    }
  }

  // Bellman correction: the inner solver sets boundary value = 0, but
  // boundary states should have value V(s). Re-sweep non-boundary states
  // to propagate correct boundary costs.
  for (std::size_t sweep = 0; sweep < _current_subssp_states.size(); ++sweep) {
    double max_residual = 0.0;
    for (const auto &st : _current_subssp_states) {
      if (_boundary_states.find(st) != _boundary_states.end()) {
        continue;
      }
      auto pit = _policy.find(st);
      if (pit == _policy.end())
        continue;
      double qval = 0.0;
      bool cost_added = false;
      auto next_dist =
          _domain.get_next_state_distribution(st, pit->second).get_values();
      for (auto ns : next_dist) {
        if (!cost_added) {
          qval +=
              _domain.get_transition_value(st, pit->second, ns.state()).cost();
          cost_added = true;
        }
        double ns_val;
        if (_boundary_states.find(ns.state()) != _boundary_states.end()) {
          ns_val = get_value(ns.state());
        } else {
          auto vit = _value_function.find(ns.state());
          ns_val = (vit != _value_function.end()) ? vit->second
                                                  : get_value(ns.state());
        }
        qval += ns.probability() * _discount * ns_val;
      }
      double residual = std::abs(qval - get_value(st));
      _value_function[st] = qval;
      if (residual > max_residual)
        max_residual = residual;
    }
    if (max_residual < _epsilon)
      break;
  }

  // Update boundary state values from global V
  for (const auto &st : _boundary_states) {
    if (!_goal_checker(_domain, st)) {
      _value_function[st] = get_value(st);
    } else {
      _value_function[st] = 0.0;
    }
  }
}

// --- Main solve loop (Algorithm 2) ---

SK_SSIPP_TEMPLATE_DECL
void SK_SSIPP_CLASS::solve(const State &s) {
  try {
    Logger::info("Running SSiPP solver with depth " +
                 StringConverter::from(_depth));
    _start_time = std::chrono::high_resolution_clock::now();

    State current = s;
    _nb_sub_ssps = 0;

    while (!_goal_checker(_domain, current) && _nb_sub_ssps < _max_iterations) {
      _nb_sub_ssps++;

      if (_verbose) {
        Logger::debug("SSiPP: iteration " +
                      StringConverter::from(_nb_sub_ssps));
      }

      build_short_sighted_ssp(current);
      solve_subssp(current);

      if (_callback(*this, _domain)) {
        break;
      }

      bool reached_boundary = false;
      while (!reached_boundary) {
        if (_boundary_states.find(current) != _boundary_states.end()) {
          if (_goal_checker(_domain, current)) {
            reached_boundary = true;
            break;
          }
          reached_boundary = true;
          break;
        }

        auto pit = _policy.find(current);
        if (pit == _policy.end()) {
          break;
        }

        auto next_dist =
            _domain.get_next_state_distribution(current, pit->second)
                .get_values();
        std::vector<double> weights;
        std::vector<State> states;
        for (auto ns : next_dist) {
          states.push_back(ns.state());
          weights.push_back(ns.probability());
        }
        if (states.empty())
          break;

        if (states.size() == 1) {
          current = states[0];
        } else {
          std::random_device rd;
          std::mt19937 gen(rd());
          std::discrete_distribution<> dist(weights.begin(), weights.end());
          current = states[dist(gen)];
        }
      }

      if (!reached_boundary) {
        break;
      }

      if (_goal_checker(_domain, current)) {
        break;
      }
    }

    Logger::info("SSiPP finished in " +
                 StringConverter::from((double)get_solving_time() / 1e3) +
                 " seconds with " + StringConverter::from(_nb_sub_ssps) +
                 " sub-SSP iterations and " +
                 StringConverter::from(_value_function.size()) +
                 " explored states.");
  } catch (const std::exception &e) {
    Logger::error("SSiPP failed: " + std::string(e.what()));
    throw;
  }
}

// --- Policy query ---

SK_SSIPP_TEMPLATE_DECL
bool SK_SSIPP_CLASS::is_solution_defined_for(const State &s) {
  if (_policy.find(s) != _policy.end() || _goal_checker(_domain, s)) {
    return true;
  }
  solve(s);
  return _policy.find(s) != _policy.end() || _goal_checker(_domain, s);
}

SK_SSIPP_TEMPLATE_DECL
const typename SK_SSIPP_CLASS::Action &
SK_SSIPP_CLASS::get_best_action(const State &s) {
  auto it = _policy.find(s);
  if (it != _policy.end()) {
    return it->second;
  }
  solve(s);
  it = _policy.find(s);
  if (it != _policy.end()) {
    return it->second;
  }
  throw std::runtime_error(
      "SKDECIDE exception: no best action found in SSiPP policy.");
}

SK_SSIPP_TEMPLATE_DECL
typename SK_SSIPP_CLASS::Value
SK_SSIPP_CLASS::get_best_value(const State &s) const {
  Value val;
  val.cost(get_value(s));
  return val;
}

// --- Statistics ---

SK_SSIPP_TEMPLATE_DECL
std::size_t SK_SSIPP_CLASS::get_nb_explored_states() const {
  return _value_function.size();
}

SK_SSIPP_TEMPLATE_DECL
std::size_t SK_SSIPP_CLASS::get_nb_sub_ssps() const { return _nb_sub_ssps; }

SK_SSIPP_TEMPLATE_DECL
std::size_t SK_SSIPP_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

// --- State set accessors ---

SK_SSIPP_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_SSIPP_CLASS::State>::Set
SK_SSIPP_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set result;
  for (const auto &entry : _value_function) {
    result.insert(entry.first);
  }
  return result;
}

SK_SSIPP_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_SSIPP_CLASS::State>::Set
SK_SSIPP_CLASS::get_current_subssp_states() const {
  return _current_subssp_states;
}

SK_SSIPP_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_SSIPP_CLASS::State>::Set
SK_SSIPP_CLASS::get_boundary_states() const {
  return _boundary_states;
}

} // namespace skdecide

#endif // SKDECIDE_SSIPP_IMPL_HH
