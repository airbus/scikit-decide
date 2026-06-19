/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Implementation of FRET (Algorithm 1) from:
 * Kolobov, Mausam, Weld, Geffner, "Heuristic Search for Generalized
 * Stochastic Shortest Path MDPs", ICAPS 2011.
 */
#ifndef SKDECIDE_FRET_IMPL_HH
#define SKDECIDE_FRET_IMPL_HH

#include <algorithm>
#include <cmath>
#include <limits>
#include <stack>

#include "utils/logging.hh"
#include "utils/string_converter.hh"

#include "hub/solver/inner_solver/inner_solver_registry.hh"
#include "hub/solver/inner_solver/meta_inner_solver.hh"

namespace skdecide {

#define SK_FRET_TEMPLATE_DECL                                                  \
  template <typename Tdomain, typename Texecution_policy,                      \
            template <typename, typename> class TinnerSolver>

#define SK_FRET_CLASS FRETSolver<Tdomain, Texecution_policy, TinnerSolver>

// --- Constructor ---

SK_FRET_TEMPLATE_DECL
template <typename... InnerSolverArgs>
SK_FRET_CLASS::FRETSolver(Domain &domain,
                          const GoalCheckerFunctor &goal_checker,
                          const HeuristicFunctor &heuristic, double discount,
                          double epsilon, double dead_end_cost,
                          const CallbackFunctor &callback, bool verbose,
                          InnerSolverArgs &&...inner_solver_args)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _discount(discount), _epsilon(epsilon), _dead_end_cost(dead_end_cost),
      _callback(callback), _verbose(verbose), _nb_fret_iterations(0),
      _nb_traps_eliminated(0) {

  auto captured_args =
      std::make_tuple(std::forward<InnerSolverArgs>(inner_solver_args)...);

  _inner_solver_factory = [this, captured_args = std::move(captured_args)](
                              Domain &d, GoalCheckerFunctor sub_gc,
                              HeuristicFunctor sub_h) mutable {
    return std::apply(
        [&](auto &&tv_orig, auto &&...rest) {
          auto dynamic_tv = [this](const State &st) -> Value {
            if (_dead_end_states.find(st) != _dead_end_states.end()) {
              Value v;
              v.cost(_dead_end_cost);
              return v;
            }
            return Value(0.0, false);
          };
          return std::make_unique<InnerSolver>(
              d, sub_gc, sub_h, dynamic_tv,
              std::forward<decltype(rest)>(rest)...);
        },
        captured_args);
  };

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm FRET");
  }
}

SK_FRET_TEMPLATE_DECL
void SK_FRET_CLASS::clear() {
  _value_function.clear();
  _policy.clear();
  _dead_end_states.clear();
  _trapped_sccs.clear();
  _nb_fret_iterations = 0;
  _nb_traps_eliminated = 0;
}

SK_FRET_TEMPLATE_DECL
double SK_FRET_CLASS::get_value(const State &s) const {
  auto it = _value_function.find(s);
  if (it != _value_function.end()) {
    return it->second;
  }
  return _heuristic(_domain, s).cost();
}

// --- Build greedy graph ---

SK_FRET_TEMPLATE_DECL
void SK_FRET_CLASS::build_greedy_graph(InnerSolver &inner, GreedyGraph &graph) {
  graph.clear();
  auto explored = inner.get_explored_states();

  for (const auto &s : explored) {
    if (!inner.is_solution_defined_for(s))
      continue;
    if (_goal_checker(_domain, s))
      continue;
    if (_dead_end_states.find(s) != _dead_end_states.end())
      continue;

    double v_s = inner.get_best_value(s).cost();
    auto actions = _domain.get_applicable_actions(s).get_elements();

    for (auto a : actions) {
      double qval = 0.0;
      bool cost_added = false;
      std::vector<std::pair<State, double>> successors;

      auto next_dist = _domain.get_next_state_distribution(s, a).get_values();
      for (auto ns : next_dist) {
        if (!cost_added) {
          qval += _domain.get_transition_value(s, a, ns.state()).cost();
          cost_added = true;
        }
        double ns_val;
        if (inner.is_solution_defined_for(ns.state())) {
          ns_val = inner.get_best_value(ns.state()).cost();
        } else {
          ns_val = get_value(ns.state());
        }
        qval += ns.probability() * _discount * ns_val;
        successors.push_back(std::make_pair(ns.state(), ns.probability()));
      }

      if (qval <= v_s + _epsilon) {
        auto &adj = graph[s];
        for (const auto &succ : successors) {
          if (succ.second > 0.0 &&
              _dead_end_states.find(succ.first) == _dead_end_states.end() &&
              (inner.is_solution_defined_for(succ.first) ||
               _goal_checker(_domain, succ.first) ||
               _domain.is_terminal(succ.first))) {
            adj.push_back(succ.first);
          }
        }
      }
    }

    if (graph.find(s) == graph.end()) {
      graph[s] = {};
    }
  }

  // Add explored terminal non-goal states as nodes with no outgoing edges.
  // These are dead-end candidates that FRET should detect as permanent traps.
  for (const auto &s : explored) {
    if (_goal_checker(_domain, s))
      continue;
    if (_dead_end_states.find(s) != _dead_end_states.end())
      continue;
    if (inner.is_solution_defined_for(s))
      continue;
    if (!_domain.is_terminal(s))
      continue;
    if (graph.find(s) == graph.end()) {
      graph[s] = {};
    }
  }
}

// --- Tarjan SCC ---

SK_FRET_TEMPLATE_DECL
std::vector<std::vector<typename SK_FRET_CLASS::State>>
SK_FRET_CLASS::tarjan_scc(const GreedyGraph &graph) {
  std::vector<std::vector<State>> sccs;
  using TarjanDataMap = typename MapTypeDeducer<State, TarjanData>::Map;
  using StateVecConstIter = typename std::vector<State>::const_iterator;

  TarjanDataMap data;
  std::stack<State> tarjan_stack;
  std::size_t index = 0;

  struct Frame {
    State node;
    StateVecConstIter neighbor_it;
    StateVecConstIter neighbor_end;
  };
  std::stack<Frame> call_stack;

  for (const auto &entry : graph) {
    const State &root = entry.first;
    if (data.find(root) != data.end())
      continue;

    data[root] = {index, index, true};
    index++;
    tarjan_stack.push(root);

    auto root_it = graph.find(root);
    call_stack.push({root, root_it->second.cbegin(), root_it->second.cend()});

    while (!call_stack.empty()) {
      Frame &frame = call_stack.top();
      bool pushed = false;

      while (frame.neighbor_it != frame.neighbor_end) {
        const State &w = *frame.neighbor_it;
        ++frame.neighbor_it;

        if (data.find(w) == data.end()) {
          data[w] = {index, index, true};
          index++;
          tarjan_stack.push(w);

          auto w_it = graph.find(w);
          if (w_it != graph.end()) {
            call_stack.push({w, w_it->second.cbegin(), w_it->second.cend()});
          } else {
            call_stack.push({w, {}, {}});
          }
          pushed = true;
          break;
        } else if (data[w].on_stack) {
          auto &d_v = data[frame.node];
          if (data[w].idx < d_v.low) {
            d_v.low = data[w].idx;
          }
        }
      }

      if (!pushed) {
        State v = frame.node;
        call_stack.pop();

        if (!call_stack.empty()) {
          auto &d_parent = data[call_stack.top().node];
          if (data[v].low < d_parent.low) {
            d_parent.low = data[v].low;
          }
        }

        if (data[v].low == data[v].idx) {
          std::vector<State> scc;
          while (true) {
            State w = tarjan_stack.top();
            tarjan_stack.pop();
            data[w].on_stack = false;
            scc.push_back(w);
            if (typename State::Equal()(w, v))
              break;
          }
          sccs.push_back(std::move(scc));
        }
      }
    }
  }

  return sccs;
}

// --- Eliminate traps ---

SK_FRET_TEMPLATE_DECL
bool SK_FRET_CLASS::eliminate_traps(InnerSolver &inner) {
  GreedyGraph greedy_graph;
  build_greedy_graph(inner, greedy_graph);

  auto sccs = tarjan_scc(greedy_graph);

  _trapped_sccs.clear();
  bool found_trap = false;

  for (const auto &scc : sccs) {
    if (scc.size() == 0)
      continue;

    bool has_goal = false;
    for (const auto &s : scc) {
      if (_goal_checker(_domain, s)) {
        has_goal = true;
        break;
      }
    }
    if (has_goal)
      continue;

    typename SetTypeDeducer<State>::Set scc_set;
    for (const auto &s : scc) {
      scc_set.insert(s);
    }

    bool has_outgoing = false;
    for (const auto &s : scc) {
      auto git = greedy_graph.find(s);
      if (git == greedy_graph.end())
        continue;
      for (const auto &succ : git->second) {
        if (scc_set.find(succ) == scc_set.end()) {
          has_outgoing = true;
          break;
        }
      }
      if (has_outgoing)
        break;
    }
    if (has_outgoing)
      continue;

    // This SCC is a trap (no outgoing edges, no goals)
    found_trap = true;
    _nb_traps_eliminated++;

    typename SetTypeDeducer<State>::Set trap_set;
    for (const auto &s : scc) {
      trap_set.insert(s);
    }
    _trapped_sccs.push_back(trap_set);

    bool has_exit_action = false;
    double best_exit_qval = std::numeric_limits<double>::infinity();

    for (const auto &s : scc) {
      auto actions = _domain.get_applicable_actions(s).get_elements();
      for (auto a : actions) {
        bool exits = false;
        double qval = 0.0;
        bool cost_added = false;

        auto next_dist = _domain.get_next_state_distribution(s, a).get_values();
        for (auto ns : next_dist) {
          if (!cost_added) {
            qval += _domain.get_transition_value(s, a, ns.state()).cost();
            cost_added = true;
          }
          double ns_val;
          if (inner.is_solution_defined_for(ns.state())) {
            ns_val = inner.get_best_value(ns.state()).cost();
          } else {
            ns_val = get_value(ns.state());
          }
          qval += ns.probability() * _discount * ns_val;

          if (scc_set.find(ns.state()) == scc_set.end()) {
            exits = true;
          }
        }

        if (exits) {
          has_exit_action = true;
          if (qval < best_exit_qval) {
            best_exit_qval = qval;
          }
        }
      }
    }

    if (!has_exit_action) {
      // Permanent trap (dead end)
      for (const auto &s : scc) {
        _value_function[s] = _dead_end_cost;
        _dead_end_states.insert(s);
      }
      if (_verbose) {
        Logger::debug("FRET: permanent trap (dead end) with " +
                      StringConverter::from(scc.size()) + " states");
      }
    } else {
      // Transient trap
      for (const auto &s : scc) {
        _value_function[s] = best_exit_qval;
      }
      if (_verbose) {
        Logger::debug(
            "FRET: transient trap with " + StringConverter::from(scc.size()) +
            " states, exit Q-value = " + StringConverter::from(best_exit_qval));
      }
    }
  }

  return found_trap;
}

// --- Extract proper policy (lines 18-28 of Algorithm 1) ---

SK_FRET_TEMPLATE_DECL
void SK_FRET_CLASS::extract_proper_policy(InnerSolver &inner, const State &s) {
  _policy.clear();
  typename SetTypeDeducer<State>::Set processed;

  // Start with goal states
  auto explored = inner.get_explored_states();
  for (const auto &st : explored) {
    if (_goal_checker(_domain, st)) {
      processed.insert(st);
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &st : explored) {
      if (processed.find(st) != processed.end())
        continue;
      if (_dead_end_states.find(st) != _dead_end_states.end())
        continue;

      auto actions = _domain.get_applicable_actions(st).get_elements();
      for (auto a : actions) {
        double qval = 0.0;
        bool cost_added = false;
        bool reaches_processed = false;

        auto next_dist =
            _domain.get_next_state_distribution(st, a).get_values();
        for (auto ns : next_dist) {
          if (!cost_added) {
            qval += _domain.get_transition_value(st, a, ns.state()).cost();
            cost_added = true;
          }
          double ns_val;
          if (inner.is_solution_defined_for(ns.state())) {
            ns_val = inner.get_best_value(ns.state()).cost();
          } else {
            ns_val = get_value(ns.state());
          }
          qval += ns.probability() * _discount * ns_val;

          if (processed.find(ns.state()) != processed.end()) {
            reaches_processed = true;
          }
        }

        double v_s = inner.is_solution_defined_for(st)
                         ? inner.get_best_value(st).cost()
                         : get_value(st);

        if (reaches_processed && qval <= v_s + _epsilon) {
          _policy[st] = a;
          _value_function[st] = v_s;
          processed.insert(st);
          changed = true;
          break;
        }
      }
    }
  }
}

// --- Main solve loop (Algorithm 1) ---

SK_FRET_TEMPLATE_DECL
void SK_FRET_CLASS::solve(const State &s) {
  try {
    Logger::info("Running FRET solver");
    _start_time = std::chrono::high_resolution_clock::now();
    _nb_fret_iterations = 0;

    while (true) {
      _nb_fret_iterations++;

      if (_verbose) {
        Logger::debug("FRET: iteration " +
                      StringConverter::from(_nb_fret_iterations));
      }

      // Find-and-Revise: create inner solver with current V as heuristic
      auto fret_heuristic = [this](Domain &d, const State &st) -> Value {
        Value v;
        v.cost(get_value(st));
        return v;
      };

      auto inner =
          _inner_solver_factory(_domain, _goal_checker, fret_heuristic);
      inner->solve(s);

      // Extract values from inner solver (skip known dead ends)
      auto explored = inner->get_explored_states();
      for (const auto &st : explored) {
        if (_dead_end_states.find(st) != _dead_end_states.end())
          continue;
        if (inner->is_solution_defined_for(st)) {
          _value_function[st] = inner->get_best_value(st).cost();
        }
      }

      // Eliminate traps
      bool traps_found = eliminate_traps(*inner);

      if (_callback(*this, _domain)) {
        break;
      }

      if (!traps_found) {
        // No traps → converged
        extract_proper_policy(*inner, s);
        break;
      }
    }

    Logger::info(
        "FRET finished in " +
        StringConverter::from((double)get_solving_time() / 1e3) +
        " seconds with " + StringConverter::from(_nb_fret_iterations) +
        " iterations and " + StringConverter::from(_nb_traps_eliminated) +
        " traps eliminated, " + StringConverter::from(_value_function.size()) +
        " explored states.");
  } catch (const std::exception &e) {
    Logger::error("FRET failed: " + std::string(e.what()));
    throw;
  }
}

// --- Policy query ---

SK_FRET_TEMPLATE_DECL
bool SK_FRET_CLASS::is_solution_defined_for(const State &s) const {
  return _policy.find(s) != _policy.end() || _goal_checker(_domain, s);
}

SK_FRET_TEMPLATE_DECL
const typename SK_FRET_CLASS::Action &
SK_FRET_CLASS::get_best_action(const State &s) const {
  auto it = _policy.find(s);
  if (it != _policy.end()) {
    return it->second;
  }
  throw std::runtime_error(
      "SKDECIDE exception: no best action found in FRET policy.");
}

SK_FRET_TEMPLATE_DECL
typename SK_FRET_CLASS::Value
SK_FRET_CLASS::get_best_value(const State &s) const {
  Value val;
  val.cost(get_value(s));
  return val;
}

// --- Statistics ---

SK_FRET_TEMPLATE_DECL
std::size_t SK_FRET_CLASS::get_nb_explored_states() const {
  return _value_function.size();
}

SK_FRET_TEMPLATE_DECL
std::size_t SK_FRET_CLASS::get_nb_fret_iterations() const {
  return _nb_fret_iterations;
}

SK_FRET_TEMPLATE_DECL
std::size_t SK_FRET_CLASS::get_nb_traps_eliminated() const {
  return _nb_traps_eliminated;
}

SK_FRET_TEMPLATE_DECL
std::size_t SK_FRET_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

// --- State set accessors ---

SK_FRET_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_FRET_CLASS::State>::Set
SK_FRET_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set result;
  for (const auto &entry : _value_function) {
    result.insert(entry.first);
  }
  return result;
}

SK_FRET_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_FRET_CLASS::State>::Set
SK_FRET_CLASS::get_dead_end_states() const {
  return _dead_end_states;
}

SK_FRET_TEMPLATE_DECL
std::vector<typename SetTypeDeducer<typename SK_FRET_CLASS::State>::Set>
SK_FRET_CLASS::get_trapped_sccs() const {
  return _trapped_sccs;
}

SK_FRET_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_FRET_CLASS> SK_FRET_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> goal_checker,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> /*terminal_value*/,
    const Params &params, bool verbose) {
  std::string inner_name =
      params.template get<std::string>("inner_solver", std::string("LRTDP"));

  auto fret_factory =
      [inner_name, params, verbose](
          Domain &dd, std::function<Predicate(Domain &, const State &)> sub_gc,
          std::function<Value(Domain &, const State &)> sub_h,
          std::function<Value(const State &)> sub_tv)
      -> std::unique_ptr<MetaInnerSolverBase<Domain>> {
    const auto &inner_entry =
        find_inner_solver<Domain, ExecutionPolicy>(inner_name);
    return inner_entry.create(dd, sub_gc, sub_h, sub_tv, params, verbose);
  };

  auto dummy_tv = [](const State &) -> Value { return Value(); };

  return std::make_unique<FRETSolver>(
      domain, goal_checker, heuristic,
      params.template get<double>("discount", 1.0),
      params.template get<double>("epsilon", 0.001),
      params.template get<double>("dead_end_cost", 10000.0),
      CallbackFunctor([](const FRETSolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose), dummy_tv,
      std::move(fret_factory));
}

} // namespace skdecide

#endif // SKDECIDE_FRET_IMPL_HH
