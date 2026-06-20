/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_GPCI_IMPL_HH
#define SKDECIDE_GPCI_IMPL_HH

#include <cmath>
#include <limits>
#include <queue>

#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_GPCI_SOLVER_TEMPLATE_DECL                                           \
  template <typename Tdomain, typename Texecution_policy>

#define SK_GPCI_SOLVER_CLASS GPCISolver<Tdomain, Texecution_policy>

SK_GPCI_SOLVER_TEMPLATE_DECL
SK_GPCI_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr), goal_probability(0.0), goal_cost(0.0),
      terminal(false), goal(false) {}

SK_GPCI_SOLVER_TEMPLATE_DECL
const typename SK_GPCI_SOLVER_CLASS::State &
SK_GPCI_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
SK_GPCI_SOLVER_CLASS::ActionNode::ActionNode(const Action &a) : action(a) {}

SK_GPCI_SOLVER_TEMPLATE_DECL
SK_GPCI_SOLVER_CLASS::GPCISolver(Domain &domain,
                                 const GoalCheckerFunctor &goal_checker,
                                 double epsilon,
                                 const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _epsilon(epsilon),
      _callback(callback), _verbose(verbose),
      _current_phase(Phase::ENUMERATION), _nb_prob_iterations(0),
      _nb_cost_iterations(0) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm GPCI");
  }
}

SK_GPCI_SOLVER_TEMPLATE_DECL
void SK_GPCI_SOLVER_CLASS::clear() {
  _graph.clear();
  _non_goal_states.clear();
  _nb_prob_iterations = 0;
  _nb_cost_iterations = 0;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
void SK_GPCI_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " GPCI solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    _current_phase = Phase::ENUMERATION;
    enumerate_reachable_states(s);
    Logger::info("GPCI: enumerated " + StringConverter::from(_graph.size()) +
                 " reachable states (" +
                 StringConverter::from(_non_goal_states.size()) + " non-goal)");

    // Phase 1: goal-probability iteration (eq. 5)
    _current_phase = Phase::PROBABILITY;
    _nb_prob_iterations = 0;
    bool converged = false;

    while (!converged && !_callback(*this, _domain)) {
      _nb_prob_iterations++;
      double max_residual = 0.0;

      std::for_each(ExecutionPolicy::policy, _non_goal_states.begin(),
                    _non_goal_states.end(),
                    [this, &max_residual](StateNode *sn) {
                      double residual = probability_update(*sn);
                      _execution_policy.protect([&max_residual, &residual] {
                        max_residual = std::max(max_residual, residual);
                      });
                    });

      converged = (max_residual < _epsilon);

      if (_verbose) {
        Logger::debug(
            "GPCI phase 1 (probability): iteration " +
            StringConverter::from(_nb_prob_iterations) +
            ", max residual = " + StringConverter::from(max_residual));
      }
    }

    Logger::info("GPCI phase 1 converged in " +
                 StringConverter::from(_nb_prob_iterations) + " iterations");

    // Build list of states with P*(s) > epsilon for phase 2
    std::vector<StateNode *> reachable_states;
    for (auto *sn : _non_goal_states) {
      if (sn->goal_probability > _epsilon) {
        reachable_states.push_back(sn);
      }
    }

    Logger::info("GPCI: " + StringConverter::from(reachable_states.size()) +
                 " states with P*(s) > 0 for phase 2");

    // Phase 2: goal-cost iteration (eq. 6)
    _current_phase = Phase::COST;
    _nb_cost_iterations = 0;
    converged = false;

    while (!converged && !_callback(*this, _domain)) {
      _nb_cost_iterations++;
      double max_residual = 0.0;

      std::for_each(ExecutionPolicy::policy, reachable_states.begin(),
                    reachable_states.end(),
                    [this, &max_residual](StateNode *sn) {
                      double residual = cost_update(*sn);
                      _execution_policy.protect([&max_residual, &residual] {
                        max_residual = std::max(max_residual, residual);
                      });
                    });

      converged = (max_residual < _epsilon);

      if (_verbose) {
        Logger::debug(
            "GPCI phase 2 (cost): iteration " +
            StringConverter::from(_nb_cost_iterations) +
            ", max residual = " + StringConverter::from(max_residual));
      }
    }

    Logger::info(
        "GPCI finished from state " + s.print() + " in " +
        StringConverter::from(_nb_prob_iterations) + " + " +
        StringConverter::from(_nb_cost_iterations) + " iterations and " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds.");
  } catch (const std::exception &e) {
    Logger::error("GPCI failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_GPCI_SOLVER_TEMPLATE_DECL
void SK_GPCI_SOLVER_CLASS::enumerate_reachable_states(const State &s) {
  if (_verbose)
    Logger::debug("Enumerating reachable states from " + s.print());

  std::queue<StateNode *> frontier;

  auto si = _graph.emplace(s);
  StateNode &root = const_cast<StateNode &>(*(si.first));
  if (si.second) {
    if (_goal_checker(_domain, s)) {
      root.goal = true;
      root.terminal = true;
      root.goal_probability = 1.0;
    } else if (_domain.is_terminal(s)) {
      root.terminal = true;
    } else {
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

  _non_goal_states.clear();
  for (auto &sn : _graph) {
    StateNode &node = const_cast<StateNode &>(sn);
    if (!node.goal && !node.terminal) {
      _non_goal_states.push_back(&node);
    }
  }
}

SK_GPCI_SOLVER_TEMPLATE_DECL
void SK_GPCI_SOLVER_CLASS::expand(StateNode &s) {
  if (_verbose)
    Logger::debug("Expanding state " + s.state.print());

  auto applicable_actions =
      _domain.get_applicable_actions(s.state).get_elements();

  std::for_each(
      ExecutionPolicy::policy, applicable_actions.begin(),
      applicable_actions.end(), [this, &s](auto a) {
        _execution_policy.protect(
            [&s, &a] { s.actions.push_back(std::make_unique<ActionNode>(a)); });
        ActionNode &an = *(s.actions.back());
        auto next_states =
            _domain.get_next_state_distribution(s.state, a).get_values();

        for (auto ns : next_states) {
          std::pair<typename Graph::iterator, bool> i;
          _execution_policy.protect(
              [this, &i, &ns] { i = _graph.emplace(ns.state()); });
          StateNode &next_node = const_cast<StateNode &>(*(i.first));
          an.outcomes.push_back(std::make_tuple(
              ns.probability(),
              _domain.get_transition_value(s.state, a, next_node.state).cost(),
              &next_node));

          if (i.second) {
            if (_goal_checker(_domain, next_node.state)) {
              next_node.goal = true;
              next_node.terminal = true;
              next_node.goal_probability = 1.0;
            } else if (_domain.is_terminal(next_node.state)) {
              next_node.terminal = true;
            }
          }
        }
      });
}

// Phase 1: P*_n(s) = max_a Σ T(s,a,s') P*_{n-1}(s')
SK_GPCI_SOLVER_TEMPLATE_DECL
double SK_GPCI_SOLVER_CLASS::probability_update(StateNode &s) {
  double old_prob = s.goal_probability;
  double best_prob = 0.0;

  for (const auto &an : s.actions) {
    double q_prob = 0.0;
    for (const auto &outcome : an->outcomes) {
      q_prob += std::get<0>(outcome) * std::get<2>(outcome)->goal_probability;
    }
    best_prob = std::max(best_prob, q_prob);
  }

  s.goal_probability = best_prob;
  return std::abs(best_prob - old_prob);
}

// Phase 2: C*_n(s) = min_{a∈A*(s)} (1/P*(s)) Σ T(s,a,s') P*(s') [c(s,a,s') +
// C*(s')]
SK_GPCI_SOLVER_TEMPLATE_DECL
double SK_GPCI_SOLVER_CLASS::cost_update(StateNode &s) {
  double old_cost = s.goal_cost;
  double p_s = s.goal_probability;

  if (p_s <= _epsilon) {
    s.goal_cost = 0.0;
    s.best_action = nullptr;
    return std::abs(old_cost);
  }

  double best_cost = std::numeric_limits<double>::infinity();
  ActionNode *best_action = nullptr;

  for (const auto &an : s.actions) {
    double action_prob = 0.0;
    for (const auto &outcome : an->outcomes) {
      action_prob +=
          std::get<0>(outcome) * std::get<2>(outcome)->goal_probability;
    }
    // Skip actions not in A*(s)
    if (std::abs(action_prob - p_s) > _epsilon)
      continue;

    double q_cost = 0.0;
    for (const auto &outcome : an->outcomes) {
      double prob = std::get<0>(outcome);
      double cost = std::get<1>(outcome);
      StateNode *next = std::get<2>(outcome);
      q_cost += prob * next->goal_probability * (cost + next->goal_cost);
    }
    q_cost /= p_s;

    if (q_cost < best_cost) {
      best_cost = q_cost;
      best_action = an.get();
    }
  }

  s.goal_cost = best_cost;
  s.best_action = best_action;
  return std::abs(best_cost - old_cost);
}

SK_GPCI_SOLVER_TEMPLATE_DECL
bool SK_GPCI_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end())
    return false;
  return si->best_action != nullptr || si->goal;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
const typename SK_GPCI_SOLVER_CLASS::Action &
SK_GPCI_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end() || si->best_action == nullptr) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_action->action;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
typename SK_GPCI_SOLVER_CLASS::Value
SK_GPCI_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    throw std::runtime_error(
        "SKDECIDE exception: state not found in GPCI graph.");
  }
  Value val;
  val.cost(si->goal_cost);
  return val;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
double SK_GPCI_SOLVER_CLASS::get_goal_probability(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    throw std::runtime_error(
        "SKDECIDE exception: state not found in GPCI graph.");
  }
  return si->goal_probability;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
double SK_GPCI_SOLVER_CLASS::get_goal_cost(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    throw std::runtime_error(
        "SKDECIDE exception: state not found in GPCI graph.");
  }
  return si->goal_cost;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
typename SK_GPCI_SOLVER_CLASS::Phase
SK_GPCI_SOLVER_CLASS::get_current_phase() const {
  return _current_phase;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
std::size_t SK_GPCI_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_GPCI_SOLVER_TEMPLATE_DECL
std::size_t SK_GPCI_SOLVER_CLASS::get_nb_prob_iterations() const {
  return _nb_prob_iterations;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
std::size_t SK_GPCI_SOLVER_CLASS::get_nb_cost_iterations() const {
  return _nb_cost_iterations;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
std::size_t SK_GPCI_SOLVER_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_GPCI_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_GPCI_SOLVER_CLASS::State>::Set
SK_GPCI_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<typename SK_GPCI_SOLVER_CLASS::State,
                        std::pair<typename SK_GPCI_SOLVER_CLASS::Action,
                                  typename SK_GPCI_SOLVER_CLASS::Value>>::Map
SK_GPCI_SOLVER_CLASS::policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (const auto &sn : _graph) {
    if (sn.best_action != nullptr) {
      Value val;
      val.cost(sn.goal_cost);
      p.insert(std::make_pair(sn.state,
                              std::make_pair(sn.best_action->action, val)));
    }
  }
  return p;
}

SK_GPCI_SOLVER_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_GPCI_SOLVER_CLASS> SK_GPCI_SOLVER_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> goal_checker,
    std::function<Value(Domain &, const State &)> /*heuristic*/,
    std::function<Value(const State &)> /*terminal_value*/,
    const Params &params, bool verbose) {
  return std::make_unique<GPCISolver>(
      domain, goal_checker, params.template get<double>("epsilon", 0.001),
      CallbackFunctor([](const GPCISolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_GPCI_IMPL_HH
