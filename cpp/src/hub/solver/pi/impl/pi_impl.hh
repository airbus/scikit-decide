/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PI_IMPL_HH
#define SKDECIDE_PI_IMPL_HH

#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === PISolver implementation ===

#define SK_PI_SOLVER_TEMPLATE_DECL                                             \
  template <typename Tdomain, typename Texecution_policy>

#define SK_PI_SOLVER_CLASS PISolver<Tdomain, Texecution_policy>

// --- StateNode ---

SK_PI_SOLVER_TEMPLATE_DECL
SK_PI_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), best_value(0.0), terminal(false),
      dead_end(false), policy_changed(false) {}

SK_PI_SOLVER_TEMPLATE_DECL
const typename SK_PI_SOLVER_CLASS::State &
SK_PI_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

// --- ActionNode ---

SK_PI_SOLVER_TEMPLATE_DECL
SK_PI_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(0.0) {}

// --- Constructor ---

SK_PI_SOLVER_TEMPLATE_DECL
SK_PI_SOLVER_CLASS::PISolver(Domain &domain, const HeuristicFunctor &heuristic,
                             const TerminalValueFunctor &terminal_value,
                             const InitialPolicyFunctor &initial_policy,
                             double discount, double epsilon,
                             std::size_t max_eval_sweeps,
                             const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _heuristic(heuristic), _terminal_value(terminal_value),
      _initial_policy(initial_policy), _discount(discount), _epsilon(epsilon),
      _max_eval_sweeps(max_eval_sweeps), _callback(callback), _verbose(verbose),
      _nb_iterations(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm Policy Iteration");
  }
}

SK_PI_SOLVER_TEMPLATE_DECL
void SK_PI_SOLVER_CLASS::clear() {
  _graph.clear();
  _non_terminal_states.clear();
  _nb_iterations = 0;
}

// --- solve ---

SK_PI_SOLVER_TEMPLATE_DECL
void SK_PI_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " Policy Iteration solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    // Phase 1: enumerate all reachable states via BFS
    enumerate_reachable_states(s);
    Logger::info("Policy Iteration: enumerated " +
                 StringConverter::from(_graph.size()) + " reachable states (" +
                 StringConverter::from(_non_terminal_states.size()) +
                 " non-terminal)");

    // Phase 2: initialize policy
    initialize_policy();

    // Phase 3: iterate policy evaluation and improvement
    _nb_iterations = 0;
    bool stable = false;

    while (!stable && !_callback(*this, _domain)) {
      _nb_iterations++;

      // Policy evaluation: Gauss-Seidel sweeps until V^pi converges
      evaluate_policy();

      // Policy improvement: greedy action selection
      stable = !improve_policy();

      if (_verbose) {
        Logger::debug("Policy Iteration: iteration " +
                      StringConverter::from(_nb_iterations) + ", policy " +
                      (stable ? "stable" : "changed"));
      }
    }

    Logger::info(
        "Policy Iteration finished from state " + s.print() + " in " +
        StringConverter::from(_nb_iterations) + " iterations and " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds.");
  } catch (const std::exception &e) {
    Logger::error("Policy Iteration failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

// --- enumerate_reachable_states (same as VI) ---

SK_PI_SOLVER_TEMPLATE_DECL
void SK_PI_SOLVER_CLASS::enumerate_reachable_states(const State &s) {
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
  } else if (root.actions.empty() && !root.terminal && !root.dead_end) {
    frontier.push(&root);
  }

  while (!frontier.empty()) {
    StateNode *current = frontier.front();
    frontier.pop();

    if (current->terminal || current->dead_end || !current->actions.empty()) {
      continue;
    }

    expand(*current);

    for (const auto &an : current->actions) {
      for (const auto &outcome : an->outcomes) {
        StateNode *ns = std::get<2>(outcome);
        if (ns->actions.empty() && !ns->terminal && !ns->dead_end) {
          frontier.push(ns);
        }
      }
    }
  }

  _non_terminal_states.clear();
  for (auto &sn : _graph) {
    StateNode &node = const_cast<StateNode &>(sn);
    if (!node.terminal && !node.dead_end) {
      _non_terminal_states.push_back(&node);
    }
  }
}

// --- expand (same as VI) ---

SK_PI_SOLVER_TEMPLATE_DECL
void SK_PI_SOLVER_CLASS::expand(StateNode &s) {
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

// --- initialize_policy ---

SK_PI_SOLVER_TEMPLATE_DECL
void SK_PI_SOLVER_CLASS::initialize_policy() {
  for (auto *sn : _non_terminal_states) {
    if (sn->dead_end || sn->actions.empty()) {
      continue;
    }

    if (_initial_policy) {
      Action target = _initial_policy(_domain, sn->state);
      sn->best_action = nullptr;
      for (const auto &an : sn->actions) {
        if (typename Action::Equal()(an->action, target)) {
          sn->best_action = an.get();
          break;
        }
      }
      if (sn->best_action == nullptr) {
        sn->best_action = sn->actions.front().get();
      }
    } else {
      sn->best_action = sn->actions.front().get();
    }
  }
}

// --- evaluate_policy (Gauss-Seidel sweeps on current policy) ---

SK_PI_SOLVER_TEMPLATE_DECL
bool SK_PI_SOLVER_CLASS::evaluate_policy() {
  bool converged = false;
  std::size_t sweep = 0;

  while (!converged) {
    sweep++;
    double max_residual = 0.0;

    std::for_each(
        ExecutionPolicy::policy, _non_terminal_states.begin(),
        _non_terminal_states.end(), [this, &max_residual](StateNode *sn) {
          if (sn->dead_end || sn->best_action == nullptr) {
            return;
          }

          double old_value = sn->best_value;
          double new_value = 0.0;

          for (const auto &outcome : sn->best_action->outcomes) {
            double prob = std::get<0>(outcome);
            double reward = std::get<1>(outcome);
            StateNode *next = std::get<2>(outcome);
            new_value += prob * (reward + _discount * next->best_value);
          }

          sn->best_value = new_value;
          double residual = std::abs(new_value - old_value);

          _execution_policy.protect([&max_residual, &residual] {
            max_residual = std::max(max_residual, residual);
          });
        });

    converged = (max_residual < _epsilon);

    if (_discount >= 1.0 && _max_eval_sweeps > 0 && sweep >= _max_eval_sweeps) {
      break;
    }
  }

  return converged;
}

// --- improve_policy (greedy action selection, returns true if changed) ---

SK_PI_SOLVER_TEMPLATE_DECL
bool SK_PI_SOLVER_CLASS::improve_policy() {
  bool any_changed = false;

  for (auto *sn : _non_terminal_states) {
    if (sn->dead_end || sn->actions.empty()) {
      continue;
    }

    sn->policy_changed = false;
    double best_value = -std::numeric_limits<double>::infinity();
    ActionNode *best_action = nullptr;

    for (const auto &an : sn->actions) {
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

    if (best_action != sn->best_action) {
      sn->best_action = best_action;
      sn->policy_changed = true;
      any_changed = true;
    }
  }

  return any_changed;
}

// --- Accessors ---

SK_PI_SOLVER_TEMPLATE_DECL
bool SK_PI_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr && !si->terminal)) {
    return false;
  }
  return true;
}

SK_PI_SOLVER_TEMPLATE_DECL
const typename SK_PI_SOLVER_CLASS::Action &
SK_PI_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_action->action;
}

SK_PI_SOLVER_TEMPLATE_DECL
typename SK_PI_SOLVER_CLASS::Value
SK_PI_SOLVER_CLASS::get_best_value(const State &s) const {
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

SK_PI_SOLVER_TEMPLATE_DECL
std::size_t SK_PI_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_PI_SOLVER_TEMPLATE_DECL
std::size_t SK_PI_SOLVER_CLASS::get_nb_iterations() const {
  return _nb_iterations;
}

SK_PI_SOLVER_TEMPLATE_DECL
std::size_t SK_PI_SOLVER_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_PI_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_PI_SOLVER_CLASS::State>::Set
SK_PI_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

SK_PI_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_PI_SOLVER_CLASS::State>::Set
SK_PI_SOLVER_CLASS::get_policy_changed_states() const {
  typename SetTypeDeducer<State>::Set changed;
  for (const auto &sn : _graph) {
    if (sn.policy_changed) {
      changed.insert(sn.state);
    }
  }
  return changed;
}

SK_PI_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<
    typename SK_PI_SOLVER_CLASS::State,
    std::pair<typename SK_PI_SOLVER_CLASS::Action, double>>::Map
SK_PI_SOLVER_CLASS::policy() const {
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
  for (const auto &sn : _graph) {
    if (sn.best_action != nullptr) {
      p.insert(std::make_pair(sn.state, std::make_pair(sn.best_action->action,
                                                       (double)sn.best_value)));
    }
  }
  return p;
}

SK_PI_SOLVER_TEMPLATE_DECL
void SK_PI_SOLVER_CLASS::set_state_dead_end(const State &s,
                                            double dead_end_cost) {
  auto si = _graph.find(s);
  if (si != _graph.end()) {
    StateNode &node = const_cast<StateNode &>(*si);
    node.dead_end = true;
    node.best_value = -dead_end_cost;
    node.best_action = nullptr;
  }
}

SK_PI_SOLVER_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_PI_SOLVER_CLASS> SK_PI_SOLVER_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> /*goal_checker*/,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> terminal_value, const Params &params,
    bool verbose) {
  return std::make_unique<PISolver>(
      domain, heuristic, terminal_value, InitialPolicyFunctor(nullptr),
      params.template get<double>("discount", 0.999),
      params.template get<double>("epsilon", 0.001),
      params.template get<std::size_t>("max_eval_sweeps", 0),
      CallbackFunctor([](const PISolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_PI_IMPL_HH
