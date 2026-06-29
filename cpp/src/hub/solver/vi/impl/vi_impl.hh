/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_VI_IMPL_HH
#define SKDECIDE_VI_IMPL_HH

#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === VISolver implementation ===

#define SK_VI_SOLVER_TEMPLATE_DECL                                             \
  template <typename Tdomain, typename Texecution_policy>

#define SK_VI_SOLVER_CLASS VISolver<Tdomain, Texecution_policy>

// --- StateNode ---

SK_VI_SOLVER_TEMPLATE_DECL
SK_VI_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), best_value(0.0), terminal(false),
      converged(false), updated_in_last_sweep(false) {}

SK_VI_SOLVER_TEMPLATE_DECL
const typename SK_VI_SOLVER_CLASS::State &
SK_VI_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

// --- ActionNode ---

SK_VI_SOLVER_TEMPLATE_DECL
SK_VI_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

// --- Constructor ---

SK_VI_SOLVER_TEMPLATE_DECL
SK_VI_SOLVER_CLASS::VISolver(Domain &domain, const HeuristicFunctor &heuristic,
                             const TerminalValueFunctor &terminal_value,
                             double discount, double epsilon,
                             std::size_t max_sweeps,
                             const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _heuristic(heuristic), _terminal_value(terminal_value),
      _discount(discount), _epsilon(epsilon), _max_sweeps(max_sweeps),
      _callback(callback), _verbose(verbose), _nb_iterations(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm Value Iteration");
  }
}

SK_VI_SOLVER_TEMPLATE_DECL
void SK_VI_SOLVER_CLASS::clear() {
  _graph.clear();
  _non_terminal_states.clear();
  _nb_iterations = 0;
}

// --- solve ---

SK_VI_SOLVER_TEMPLATE_DECL
void SK_VI_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " Value Iteration solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    // Phase 1: enumerate all reachable states via BFS
    enumerate_reachable_states(s);
    Logger::info("Value Iteration: enumerated " +
                 StringConverter::from(_graph.size()) + " reachable states (" +
                 StringConverter::from(_non_terminal_states.size()) +
                 " non-terminal)");

    // Phase 2: iterate synchronous Bellman backups (reward maximization)
    _nb_iterations = 0;
    bool converged = false;

    while (!converged && !_callback(*this, _domain)) {
      _nb_iterations++;
      double max_residual = 0.0;

      for (auto *sn : _non_terminal_states) {
        sn->updated_in_last_sweep = false;
      }

      std::for_each(ExecutionPolicy::policy, _non_terminal_states.begin(),
                    _non_terminal_states.end(),
                    [this, &max_residual](StateNode *sn) {
                      double residual = bellman_update(*sn);
                      sn->converged = (residual < _epsilon);
                      sn->updated_in_last_sweep = (residual >= _epsilon);
                      _execution_policy.protect([&max_residual, &residual] {
                        max_residual = std::max(max_residual, residual);
                      });
                    });

      converged = (max_residual < _epsilon);

      if (_discount >= 1.0 && _max_sweeps > 0 &&
          _nb_iterations >= _max_sweeps) {
        if (_verbose) {
          Logger::debug("Value Iteration: reached max_sweeps limit (" +
                        StringConverter::from(_max_sweeps) + ")");
        }
        break;
      }

      if (_verbose) {
        Logger::debug(
            "Value Iteration: iteration " +
            StringConverter::from(_nb_iterations) +
            ", max residual = " + StringConverter::from(max_residual));
      }
    }

    Logger::info(
        "Value Iteration finished from state " + s.print() + " in " +
        StringConverter::from(_nb_iterations) + " iterations and " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds.");
  } catch (const std::exception &e) {
    Logger::error("Value Iteration failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

// --- enumerate_reachable_states ---

SK_VI_SOLVER_TEMPLATE_DECL
void SK_VI_SOLVER_CLASS::enumerate_reachable_states(const State &s) {
  if (_verbose)
    Logger::debug("Enumerating reachable states from " + s.print());

  std::queue<StateNode *> frontier;

  auto si = _graph.emplace(s);
  StateNode &root = const_cast<StateNode &>(*(si.first));
  if (si.second) {
    if (_domain.is_terminal(s)) {
      root.terminal = true;
      root.best_value = _terminal_value(s).reward();
    } else {
      root.best_value = _heuristic(_domain, s).reward();
      frontier.push(&root);
    }
  } else if (root.actions.empty() && !root.terminal) {
    frontier.push(&root);
  }

  while (!frontier.empty()) {
    StateNode *current = frontier.front();
    frontier.pop();

    if (current->terminal || !current->actions.empty()) {
      continue;
    }

    expand(*current);

    for (const auto &an : current->actions) {
      for (const auto &outcome : an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        if (ns->actions.empty() && !ns->terminal) {
          frontier.push(ns);
        }
      }
    }
  }

  // Build non-terminal state ordering for iteration
  _non_terminal_states.clear();
  for (auto &sn : _graph) {
    StateNode &node = const_cast<StateNode &>(sn);
    if (!node.terminal) {
      _non_terminal_states.push_back(&node);
    }
  }
}

// --- expand ---

SK_VI_SOLVER_TEMPLATE_DECL
void SK_VI_SOLVER_CLASS::expand(StateNode &s) {
  if (_verbose)
    Logger::debug("Expanding state " + s.state.print());

  auto applicable_actions =
      _domain.get_applicable_actions(s.state).get_elements();

  std::for_each(
      ExecutionPolicy::policy, applicable_actions.begin(),
      applicable_actions.end(), [this, &s](auto a) {
        if (_verbose)
          Logger::debug("Current expanded action: " + a.print() +
                        ExecutionPolicy::print_thread());
        _execution_policy.protect(
            [&s, &a] { s.actions.push_back(std::make_unique<ActionNode>(a)); });
        ActionNode &an = *(s.actions.back());
        auto next_states =
            _domain.get_next_state_distribution(s.state, a).get_values();

        for (auto ns : next_states) {
          if (_verbose)
            Logger::debug("Current next state expansion: " +
                          ns.state().print() + ExecutionPolicy::print_thread());
          std::pair<typename Graph::iterator, bool> i;
          _execution_policy.protect(
              [this, &i, &ns] { i = _graph.emplace(ns.state()); });
          StateNode &next_node = const_cast<StateNode &>(*(i.first));
          an.outcomes.push_back(std::make_tuple(
              ns.probability(),
              _domain.get_transition_value(s.state, a, next_node.state)
                  .reward(),
              &next_node));

          if (i.second) { // new node
            if (_domain.is_terminal(next_node.state)) {
              if (_verbose)
                Logger::debug("Found terminal state " +
                              next_node.state.print() +
                              ExecutionPolicy::print_thread());
              next_node.terminal = true;
              next_node.best_value = _terminal_value(next_node.state).reward();
            } else {
              next_node.best_value =
                  _heuristic(_domain, next_node.state).reward();
              if (_verbose)
                Logger::debug("New state " + next_node.state.print() +
                              " with initial value " +
                              StringConverter::from(next_node.best_value) +
                              ExecutionPolicy::print_thread());
            }
          }
        }
      });
}

// --- bellman_update (reward maximization) ---

SK_VI_SOLVER_TEMPLATE_DECL
double SK_VI_SOLVER_CLASS::bellman_update(StateNode &s) {
  double old_value = s.best_value;
  double best_value = -std::numeric_limits<double>::infinity();
  ActionNode *best_action = nullptr;

  for (const auto &an : s.actions) {
    double q_value = 0.0;
    for (const auto &outcome : an->outcomes) {
      double prob = std::get<0>(outcome);
      double reward = std::get<1>(outcome);
      StateNode *next = std::get<2>(outcome);
      q_value += prob * (reward + _discount * next->best_value);
    }
    an->value = q_value;

    if (q_value > best_value) {
      best_value = q_value;
      best_action = an.get();
    }
  }

  s.best_value = best_value;
  s.best_action = best_action;

  return std::abs(best_value - old_value);
}

// --- Accessors ---

SK_VI_SOLVER_TEMPLATE_DECL
bool SK_VI_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr && !si->terminal)) {
    return false;
  }
  return true;
}

SK_VI_SOLVER_TEMPLATE_DECL
const typename SK_VI_SOLVER_CLASS::Action &
SK_VI_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_action->action;
}

SK_VI_SOLVER_TEMPLATE_DECL
typename SK_VI_SOLVER_CLASS::Value
SK_VI_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  Value val;
  val.reward(si->best_value);
  return val;
}

SK_VI_SOLVER_TEMPLATE_DECL
std::size_t SK_VI_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_VI_SOLVER_TEMPLATE_DECL
std::size_t SK_VI_SOLVER_CLASS::get_nb_iterations() const {
  return _nb_iterations;
}

SK_VI_SOLVER_TEMPLATE_DECL
std::size_t SK_VI_SOLVER_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_VI_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_VI_SOLVER_CLASS::State>::Set
SK_VI_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

SK_VI_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_VI_SOLVER_CLASS::State>::Set
SK_VI_SOLVER_CLASS::get_converged_states() const {
  typename SetTypeDeducer<State>::Set converged;
  for (const auto &sn : _graph) {
    if (sn.converged || sn.terminal) {
      converged.insert(sn.state);
    }
  }
  return converged;
}

SK_VI_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_VI_SOLVER_CLASS::State>::Set
SK_VI_SOLVER_CLASS::get_states_updated_in_last_sweep() const {
  typename SetTypeDeducer<State>::Set updated;
  for (const auto &sn : _graph) {
    if (sn.updated_in_last_sweep) {
      updated.insert(sn.state);
    }
  }
  return updated;
}

SK_VI_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<typename SK_VI_SOLVER_CLASS::State,
                        std::pair<typename SK_VI_SOLVER_CLASS::Action,
                                  typename SK_VI_SOLVER_CLASS::Value>>::Map
SK_VI_SOLVER_CLASS::policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (const auto &sn : _graph) {
    if (sn.best_action != nullptr) {
      Value val;
      val.reward(sn.best_value);
      p.insert(std::make_pair(sn.state,
                              std::make_pair(sn.best_action->action, val)));
    }
  }
  return p;
}

SK_VI_SOLVER_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_VI_SOLVER_CLASS> SK_VI_SOLVER_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> /*goal_checker*/,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> terminal_value, const Params &params,
    bool verbose) {
  return std::make_unique<VISolver>(
      domain, heuristic, terminal_value,
      params.template get<double>("discount", 1.0),
      params.template get<double>("epsilon", 0.001),
      params.template get<std::size_t>("max_sweeps", 0),
      CallbackFunctor([](const VISolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_VI_IMPL_HH
