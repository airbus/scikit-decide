/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_LRTDP_IMPL_HH
#define SKDECIDE_LRTDP_IMPL_HH

#include <boost/range/irange.hpp>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === LRTDPStarSolver implementation ===

#define SK_LRTDP_SOLVER_TEMPLATE_DECL                                          \
  template <typename Tdomain, typename Texecution_policy>

#define SK_LRTDP_SOLVER_CLASS LRTDPSolver<Tdomain, Texecution_policy>

SK_LRTDP_SOLVER_TEMPLATE_DECL
SK_LRTDP_SOLVER_CLASS::LRTDPSolver(
    Domain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic,
    const TerminalValueFunctor &terminal_value, bool use_labels,
    std::size_t time_budget, std::size_t rollout_budget, std::size_t max_depth,
    std::size_t residual_moving_average_window, double epsilon, double discount,
    bool online_node_garbage, const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _terminal_value(terminal_value),
      _use_terminal_value(terminal_value != nullptr), _use_labels(use_labels),
      _time_budget(time_budget), _rollout_budget(rollout_budget),
      _max_depth(max_depth),
      _residual_moving_average_window(residual_moving_average_window),
      _epsilon(epsilon), _discount(discount),
      _online_node_garbage(online_node_garbage), _callback(callback),
      _verbose(verbose), _current_state(nullptr), _nb_rollouts(0) {

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm LRTDP");
  }

  std::random_device rd;
  _gen = std::make_unique<std::mt19937>(rd());
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::clear() { _graph.clear(); }

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " LRTDP solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    auto si = _graph.emplace(s);
    StateNode &root_node = const_cast<StateNode &>(
        *(si.first)); // we won't change the real key (StateNode::state) so we
                      // are safe

    if (si.second) {
      root_node.best_value = _heuristic(_domain, s, nullptr).cost();
    }

    if (root_node.solved ||
        _goal_checker(_domain, s,
                      nullptr)) { // problem already solved from this state (was
                                  // present in _graph and already solved)
      Logger::info(
          "LRTDP finished to solve from state " + s.print() + " [" +
          ((root_node.solved && !root_node.goal) ? ("solved") : ("goal")) +
          " state]");
      return;
    }

    _nb_rollouts = 0;
    _residual_moving_average = 0.0;
    _residuals.clear();
    boost::integer_range<std::size_t> parallel_rollouts(
        0, _domain.get_parallel_capacity());

    std::for_each(ExecutionPolicy::policy, parallel_rollouts.begin(),
                  parallel_rollouts.end(),
                  [this, &root_node](const std::size_t &thread_id) {
                    do {
                      if (_verbose)
                        Logger::debug("Starting rollout " +
                                      StringConverter::from(_nb_rollouts) +
                                      ExecutionPolicy::print_thread());
                      _nb_rollouts++;
                      double root_node_record_value = root_node.best_value;
                      trial(&root_node, &thread_id);
                      update_residual_moving_average(root_node,
                                                     root_node_record_value);
                    } while (!_callback(*this, _domain, &thread_id) &&
                             (get_solving_time() < _time_budget) &&
                             (!_use_labels || !root_node.solved) &&
                             (_use_labels ||
                              ((_nb_rollouts < _rollout_budget) &&
                               (get_residual_moving_average() > _epsilon))));
                  });

    Logger::info(
        "LRTDP finished to solve from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds with " + StringConverter::from(_nb_rollouts) +
        " rollouts and visited " + StringConverter::from(_graph.size()) +
        " states. ");
  } catch (const std::exception &e) {
    Logger::error("LRTDP failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
bool SK_LRTDP_SOLVER_CLASS::is_solution_defined_for(const State &s) {
  bool res;
  _execution_policy.protect([this, &s, &res]() {
    auto si = _graph.find(s);
    if ((si == _graph.end()) || (si->best_action == nullptr)) {
      // /!\ does not mean the state is solved!
      res = false;
    } else {
      res = true;
    }
  });
  return res;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
const typename SK_LRTDP_SOLVER_CLASS::Action &
SK_LRTDP_SOLVER_CLASS::get_best_action(const State &s) {
  ActionNode *best_action = nullptr;
  _execution_policy.protect([this, &s, &best_action]() {
    auto si = _graph.find(s);
    if ((si != _graph.end()) && (si->best_action != nullptr)) {
      if (_verbose) {
        std::string str = "(";
        for (const auto &o : si->best_action->outcomes) {
          str += "\n    " + std::get<2>(o)->state.print();
        }
        str += "\n)";
        Logger::debug("Best action's outcomes:\n" + str);
      }
      if (_online_node_garbage && _current_state) {
        std::unordered_set<StateNode *> root_subgraph, child_subgraph;
        compute_reachable_subgraph(_current_state, root_subgraph);
        compute_reachable_subgraph(
            const_cast<StateNode *>(&(*si)),
            child_subgraph); // we won't change the real key (StateNode::state)
                             // so we are safe
        remove_subgraph(root_subgraph, child_subgraph);
      }
      _current_state =
          const_cast<StateNode *>(&(*si)); // we won't change the real key
                                           // (StateNode::state) so we are safe
      best_action = si->best_action;
    }
  });
  if (best_action == nullptr) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return best_action->action;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
typename SK_LRTDP_SOLVER_CLASS::Value
SK_LRTDP_SOLVER_CLASS::get_best_value(const State &s) {
  const atomic_double *best_value = nullptr;
  _execution_policy.protect([this, &s, &best_value]() {
    auto si = _graph.find(s);
    if (si != _graph.end()) {
      best_value = &(si->best_value);
    }
  });
  if (best_value == nullptr) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  Value val;
  val.cost(*best_value);
  return val;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_LRTDP_SOLVER_CLASS::get_nb_explored_states() {
  std::size_t sz = 0;
  _execution_policy.protect([this, &sz]() { sz = _graph.size(); });
  return sz;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_LRTDP_SOLVER_CLASS::get_nb_rollouts() const {
  return _nb_rollouts;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
double SK_LRTDP_SOLVER_CLASS::get_residual_moving_average() {
  double val = 0.0;
  _execution_policy.protect(
      [this, &val]() {
        if (_residuals.size() >= _residual_moving_average_window) {
          val = (double)_residual_moving_average;
        } else {
          val = std::numeric_limits<double>::infinity();
        }
      },
      _residuals_protect);
  return val;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_LRTDP_SOLVER_CLASS::get_solving_time() {
  std::size_t milliseconds_duration;
  _execution_policy.protect(
      [this, &milliseconds_duration]() {
        milliseconds_duration = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - _start_time)
                .count());
      },
      _time_mutex);
  return milliseconds_duration;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL typename MapTypeDeducer<
    typename SK_LRTDP_SOLVER_CLASS::State,
    std::pair<typename SK_LRTDP_SOLVER_CLASS::Action,
              typename SK_LRTDP_SOLVER_CLASS::Value>>::Map
SK_LRTDP_SOLVER_CLASS::get_policy() {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  _execution_policy.protect([this, &p]() {
    for (auto &n : _graph) {
      if (n.best_action != nullptr) {
        Value val;
        val.cost(n.best_value);
        p.insert(std::make_pair(n.state,
                                std::make_pair(n.best_action->action, val)));
      }
    }
  });
  return p;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::expand(StateNode *s, const std::size_t *thread_id) {
  if (_verbose)
    Logger::debug("Expanding state " + s->state.print() +
                  ExecutionPolicy::print_thread());
  auto applicable_actions =
      _domain.get_applicable_actions(s->state, thread_id).get_elements();

  // Terminal state: no applicable actions
  if (applicable_actions.empty()) {
    if (_verbose)
      Logger::debug("State " + s->state.print() +
                    " has no applicable actions (terminal)" +
                    ExecutionPolicy::print_thread());

    // Terminal states (goals or non-goals)
    if (_goal_checker(_domain, s->state, thread_id)) {
      s->goal = true;
      s->solved = true;
      // Goals always have value 0 (cost-to-go from goal is zero)
      s->best_value = 0.0;
      if (_verbose)
        Logger::debug("Terminal state " + s->state.print() + " is a GOAL" +
                      ExecutionPolicy::print_thread());
    } else {
      // Non-goal terminal (dead-end): use terminal_value if provided, else
      // heuristic Terminal states are always solved (value is fixed)
      s->best_value = _use_terminal_value
                          ? _terminal_value(s->state).cost()
                          : _heuristic(_domain, s->state, thread_id).cost();
      s->solved = true;
      if (_verbose)
        Logger::debug("Non-goal terminal state " + s->state.print() +
                      " with terminal value " +
                      StringConverter::from(s->best_value) +
                      ExecutionPolicy::print_thread());
    }
    return; // Don't expand further
  }

  for (auto a : applicable_actions) {
    if (_verbose)
      Logger::debug("Current expanded action: " + a.print() +
                    ExecutionPolicy::print_thread());
    s->actions.push_back(std::make_unique<ActionNode>(a));
    ActionNode &an = *(s->actions.back());
    auto next_states =
        _domain.get_next_state_distribution(s->state, a, thread_id)
            .get_values();
    std::vector<double> outcome_weights;

    for (auto ns : next_states) {
      std::pair<typename Graph::iterator, bool> i;
      _execution_policy.protect(
          [this, &i, &ns]() { i = _graph.emplace(ns.state()); });
      StateNode &next_node = const_cast<StateNode &>(
          *(i.first)); // we won't change the real key (StateNode::state) so
                       // we are safe
      an.outcomes.push_back(std::make_tuple(
          ns.probability(),
          _domain.get_transition_value(s->state, a, next_node.state, thread_id)
              .cost(),
          &next_node));
      outcome_weights.push_back(std::get<0>(an.outcomes.back()));
      if (_verbose)
        Logger::debug(
            "Current next state expansion: " + next_node.state.print() +
            ExecutionPolicy::print_thread());

      if (i.second) { // new node
        if (_goal_checker(_domain, next_node.state, thread_id)) {
          // Goal state: value is always 0 (cost-to-go from goal is zero)
          if (_verbose)
            Logger::debug("Found goal state " + next_node.state.print() +
                          ExecutionPolicy::print_thread());
          next_node.goal = true;
          next_node.solved = true;
          next_node.best_value = 0.0;
        } else {
          // Check if terminal: either no actions OR domain marks it terminal
          auto next_actions =
              _domain.get_applicable_actions(next_node.state, thread_id)
                  .get_elements();
          bool is_terminal = next_actions.empty() ||
                             _domain.is_terminal(next_node.state, thread_id);

          if (is_terminal) {
            // Non-goal terminal (dead-end): use terminal_value if provided,
            // else heuristic Terminal states are always solved (value is fixed)
            next_node.best_value =
                _use_terminal_value
                    ? _terminal_value(next_node.state).cost()
                    : _heuristic(_domain, next_node.state, thread_id).cost();
            next_node.solved = true;
            if (_verbose)
              Logger::debug("Found non-goal terminal state " +
                            next_node.state.print() + " with value " +
                            StringConverter::from(next_node.best_value) +
                            ExecutionPolicy::print_thread());
          } else {
            // Non-terminal: use heuristic
            next_node.best_value =
                _heuristic(_domain, next_node.state, thread_id).cost();
            if (_verbose)
              Logger::debug("New state " + next_node.state.print() +
                            " with heuristic value " +
                            StringConverter::from(next_node.best_value) +
                            ExecutionPolicy::print_thread());
          }
        }
      }
    }

    an.dist = std::discrete_distribution<>(outcome_weights.begin(),
                                           outcome_weights.end());
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
double SK_LRTDP_SOLVER_CLASS::q_value(ActionNode *a) {
  // Safety check: should never be called with nullptr
  if (a == nullptr) {
    Logger::error("SKDECIDE exception: q_value called with nullptr action");
    throw std::runtime_error(
        "SKDECIDE exception: q_value called with nullptr action");
  }

  // Accumulate into a plain double to avoid MSVC issues with
  // std::atomic<double> in arithmetic expressions (no operator+ defined;
  // implicit conversion to double may not be applied by MSVC template
  // deduction).
  double new_value = 0.0;
  for (const auto &o : a->outcomes) {
    new_value +=
        std::get<0>(o) *
        (std::get<1>(o) +
         (_discount * static_cast<double>(std::get<2>(o)->best_value)));
  }
  a->value = new_value;
  if (_verbose)
    Logger::debug("Updated Q-value of action " + a->action.print() +
                  " with value " + StringConverter::from(a->value) +
                  ExecutionPolicy::print_thread());
  return new_value;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
typename SK_LRTDP_SOLVER_CLASS::ActionNode *
SK_LRTDP_SOLVER_CLASS::greedy_action(StateNode *s,
                                     const std::size_t *thread_id) {
  double best_value = std::numeric_limits<double>::infinity();
  ActionNode *best_action = nullptr;

  if (s->actions.empty()) {
    expand(s, thread_id);
  }

  for (auto &a : s->actions) {
    if (q_value(a.get()) < best_value) {
      best_value = static_cast<double>(a->value);
      best_action = a.get();
    }
  }

  if (_verbose) {
    if (best_action != nullptr) {
      Logger::debug("Greedy action of state " + s->state.print() + ": " +
                    best_action->action.print() + " with value " +
                    StringConverter::from(best_value) +
                    ExecutionPolicy::print_thread());
    } else {
      Logger::debug("Greedy action of state " + s->state.print() +
                    ": NONE (dead-end)" + ExecutionPolicy::print_thread());
    }
  }

  return best_action;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::update(StateNode *s, const std::size_t *thread_id) {
  if (_verbose)
    Logger::debug("Updating state " + s->state.print() +
                  ExecutionPolicy::print_thread());
  s->best_action = greedy_action(s, thread_id);

  // Dead-end state: no best action (nullptr returned from greedy_action)
  if (s->best_action == nullptr) {
    // Value was already set in expand() to infinity (or terminal_value for
    // goals/terminals) Don't update it here
    if (_verbose)
      Logger::debug("State " + s->state.print() +
                    " has no actions (dead-end), keeping value " +
                    StringConverter::from(s->best_value) +
                    ExecutionPolicy::print_thread());
    return;
  }

  s->best_value = (double)s->best_action->value;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
typename SK_LRTDP_SOLVER_CLASS::StateNode *
SK_LRTDP_SOLVER_CLASS::pick_next_state(ActionNode *a) {
  // Safety check: should never be called with nullptr
  if (a == nullptr) {
    Logger::error(
        "SKDECIDE exception: pick_next_state called with nullptr action");
    throw std::runtime_error(
        "SKDECIDE exception: pick_next_state called with nullptr action");
  }

  StateNode *s = nullptr;
  _execution_policy.protect(
      [&a, &s, this]() {
        s = std::get<2>(a->outcomes[a->dist(*_gen)]);
        if (_verbose)
          Logger::debug("Picked next state " + s->state.print() +
                        " from action " + a->action.print() +
                        ExecutionPolicy::print_thread());
      },
      _gen_mutex);
  return s;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
double SK_LRTDP_SOLVER_CLASS::residual(StateNode *s,
                                       const std::size_t *thread_id) {
  s->best_action = greedy_action(s, thread_id);

  // Dead-end state: no applicable actions
  if (s->best_action == nullptr) {
    if (_verbose)
      Logger::debug("State " + s->state.print() +
                    " is a DEAD-END (no applicable actions)" +
                    ExecutionPolicy::print_thread());
    // Dead-end states are considered converged (infinite residual would prevent
    // labeling) The terminal_value should have been set during expansion
    return 0.0; // Residual is 0 since value is stable (terminal)
  }

  // States where all actions lead to dead-ends (infinite Q-values)
  // Both the value and best action value converge to infinity
  // Explicit casts to double required: MSVC's std::isfinite/std::fabs are
  // function templates that deduce _Ty = std::atomic<double> and try to
  // pass by value, which triggers the deleted copy constructor.
  // static_cast<double> forces operator double() before template deduction.
  if (!std::isfinite(static_cast<double>(s->best_action->value))) {
    if (!std::isfinite(static_cast<double>(s->best_value))) {
      // Both infinite: converged to dead-end state value
      return 0.0;
    }
    // Value still finite but best action is infinite: not converged yet
    // This happens during initial exploration before value propagates
    return std::numeric_limits<double>::infinity();
  }

  double res = std::fabs(static_cast<double>(s->best_value) -
                         static_cast<double>(s->best_action->value));
  if (_verbose)
    Logger::debug("State " + s->state.print() + " has residual " +
                  StringConverter::from(res) + ExecutionPolicy::print_thread());
  return res;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
bool SK_LRTDP_SOLVER_CLASS::check_solved(StateNode *s,
                                         const std::size_t *thread_id) {
  if (_verbose) {
    _execution_policy.protect(
        [&s]() {
          Logger::debug("Checking solved status of State " + s->state.print() +
                        ExecutionPolicy::print_thread());
        },
        s->mutex);
  }

  bool rv = true;
  std::stack<StateNode *> open;
  std::stack<StateNode *> closed;
  // Paper's Algorithm 3: check if state is IN(s', open ∪ closed)
  // We track this with a single "visited" set for efficiency
  std::unordered_set<StateNode *> visited;

  if (!(s->solved)) {
    open.push(s);
    visited.insert(s); // Mark as visited when added to open
  }

  // Note: Paper's Algorithm 3 has NO depth limit in CHECKSOLVED
  // The DFS must explore the entire greedy envelope to determine if solved
  while (!open.empty() && (get_solving_time() < _time_budget)) {
    StateNode *cs = open.top();
    open.pop();
    closed.push(cs);
    // Note: cs already in visited (added when pushed to open)

    _execution_policy.protect(
        [this, &cs, &rv, &open, &visited, &thread_id]() {
          if (residual(cs, thread_id) > _epsilon) {
            rv = false;
            return;
          }

          ActionNode *a = cs->best_action; // best action updated when calling
                                           // residual(cs, thread_id)

          // Dead-end state: no applicable actions, don't expand children
          if (a == nullptr) {
            // Dead-end non-goal states should not be labeled as solved
            // (they were labeled in greedy_action if they are goals/terminals)
            if (!cs->solved) {
              rv = false; // Can't label parent states as solved if they reach
                          // an unsolved dead-end
            }
            return;
          }

          for (const auto &o : a->outcomes) {
            StateNode *ns = std::get<2>(o);
            // Paper: if ¬s'.SOLVED ∧ ¬IN(s', open ∪ closed)
            if (!(ns->solved) && (visited.find(ns) == visited.end())) {
              open.push(ns);
              visited.insert(ns); // Mark as visited when added to open
            }
          }
        },
        cs->mutex);
  }

  auto e_time = get_solving_time();
  rv = rv &&
       ((e_time < _time_budget) || ((e_time >= _time_budget) && open.empty()));

  if (rv) {
    while (!closed.empty()) {
      closed.top()->solved = true;
      closed.pop();
    }
  } else {
    while (!closed.empty() && (get_solving_time() < _time_budget)) {
      _execution_policy.protect(
          [this, &closed, &thread_id]() { update(closed.top(), thread_id); },
          closed.top()->mutex);
      closed.pop();
    }
  }

  if (_verbose) {
    _execution_policy.protect(
        [&s, &rv]() {
          Logger::debug("State " + s->state.print() + " is " +
                        (rv ? ("") : ("not")) + " solved." +
                        ExecutionPolicy::print_thread());
        },
        s->mutex);
  }

  return rv;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::trial(StateNode *s, const std::size_t *thread_id) {
  std::stack<StateNode *> visited;
  std::vector<StateNode *> current_trajectory;
  StateNode *cs = s;
  std::size_t depth = 0;
  bool found_goal = false;

  while ((!_use_labels || !(cs->solved)) && !found_goal &&
         (get_solving_time() < _time_budget) && (depth < _max_depth)) {
    depth++;
    visited.push(cs);
    current_trajectory.push_back(cs);
    _execution_policy.protect(
        [this, &cs, &found_goal, &thread_id]() {
          if (cs->goal) {
            if (_verbose)
              Logger::debug("Found goal state " + cs->state.print() +
                            ExecutionPolicy::print_thread());
            found_goal = true;
            return; // Don't continue from goal state
          }

          update(cs, thread_id);

          // Dead-end: no best action, can't pick next state
          if (cs->best_action == nullptr) {
            if (_verbose)
              Logger::debug("Trial hit dead-end state " + cs->state.print() +
                            ExecutionPolicy::print_thread());
            found_goal = true; // Treat as terminal to stop trial
            return;
          }

          cs = pick_next_state(cs->best_action);
        },
        cs->mutex);
  }

  // Save the trajectory after the trial completes (protected: multiple threads
  // run trials concurrently and the callback may read _last_trajectory)
  _execution_policy.protect(
      [this, &current_trajectory]() { _last_trajectory = current_trajectory; },
      _trajectory_mutex);

  while (_use_labels && !visited.empty() &&
         (get_solving_time() < _time_budget)) {
    cs = visited.top();
    visited.pop();

    if (!check_solved(cs, thread_id)) {
      break;
    }
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::compute_reachable_subgraph(
    StateNode *node, std::unordered_set<StateNode *> &subgraph) {
  std::unordered_set<StateNode *> frontier;
  frontier.insert(node);
  subgraph.insert(node);
  while (!frontier.empty()) {
    std::unordered_set<StateNode *> new_frontier;
    for (auto &n : frontier) {
      for (auto &action : n->actions) {
        for (auto &outcome : action->outcomes) {
          if (subgraph.find(std::get<2>(outcome)) == subgraph.end()) {
            new_frontier.insert(std::get<2>(outcome));
            subgraph.insert(std::get<2>(outcome));
          }
        }
      }
    }
    frontier = new_frontier;
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::remove_subgraph(
    std::unordered_set<StateNode *> &root_subgraph,
    std::unordered_set<StateNode *> &child_subgraph) {
  for (auto &n : root_subgraph) {
    if (child_subgraph.find(n) == child_subgraph.end()) {
      _graph.erase(StateNode(n->state));
    }
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::update_residual_moving_average(
    const StateNode &node, const double &node_record_value) {
  if (_residual_moving_average_window > 0) {
    double current_residual =
        std::fabs(node_record_value - static_cast<double>(node.best_value));
    _execution_policy.protect(
        [this, &current_residual]() {
          // Load atomic once to avoid MSVC issues with atomic arithmetic.
          double ma = static_cast<double>(_residual_moving_average);
          if (_residuals.size() < _residual_moving_average_window) {
            _residual_moving_average =
                ((ma * static_cast<double>(_residuals.size())) +
                 current_residual) /
                static_cast<double>(_residuals.size() + 1);
          } else {
            _residual_moving_average =
                ma + ((current_residual - _residuals.front()) /
                      static_cast<double>(_residual_moving_average_window));
            _residuals.pop_front();
          }
          _residuals.push_back(current_residual);
        },
        _residuals_protect);
  }
}

// === LRTDPStarSolver::StateNode implementation ===

SK_LRTDP_SOLVER_TEMPLATE_DECL
SK_LRTDP_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr),
      best_value(std::numeric_limits<double>::infinity()), goal(false),
      solved(false) {}

SK_LRTDP_SOLVER_TEMPLATE_DECL
const typename SK_LRTDP_SOLVER_CLASS::State &
SK_LRTDP_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

// === LRTDPStarSolver::ActionNode implementation ===

SK_LRTDP_SOLVER_TEMPLATE_DECL
SK_LRTDP_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(std::numeric_limits<double>::infinity()) {}

// --- State set accessors ---

SK_LRTDP_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_LRTDP_SOLVER_CLASS::State>::Set
SK_LRTDP_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored;
  for (const auto &sn : _graph) {
    explored.insert(sn.state);
  }
  return explored;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_LRTDP_SOLVER_CLASS::State>::Set
SK_LRTDP_SOLVER_CLASS::get_solved_states() const {
  typename SetTypeDeducer<State>::Set solved;
  for (const auto &sn : _graph) {
    if (sn.solved) {
      solved.insert(sn.state);
    }
  }
  return solved;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
std::vector<std::pair<typename SK_LRTDP_SOLVER_CLASS::State,
                      typename SK_LRTDP_SOLVER_CLASS::Action>>
SK_LRTDP_SOLVER_CLASS::get_last_trajectory() const {
  std::vector<std::pair<State, Action>> trajectory;
  // _execution_policy is non-const but this method is const; use the mutable
  // mutex directly (Mutex::lock/unlock are no-ops for SequentialExecution)
  _trajectory_mutex.lock();
  trajectory.reserve(_last_trajectory.size());
  for (const auto *sn : _last_trajectory) {
    Action action = sn->best_action ? sn->best_action->action : Action();
    trajectory.push_back(std::make_pair(sn->state, action));
  }
  _trajectory_mutex.unlock();
  return trajectory;
}

// === LRTAstarSolver implementation ===

#define SK_LRTASTAR_SOLVER_TEMPLATE_DECL                                       \
  template <typename Tdomain, typename Texecution_policy>

#define SK_LRTASTAR_SOLVER_CLASS LRTAstarSolver<Tdomain, Texecution_policy>

SK_LRTASTAR_SOLVER_TEMPLATE_DECL
SK_LRTASTAR_SOLVER_CLASS::LRTAstarSolver(
    Tdomain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic, std::size_t time_budget,
    std::size_t rollout_budget, std::size_t max_depth,
    const CallbackFunctor &callback, bool verbose)
    : Base(domain, goal_checker, heuristic,
           nullptr, // terminal_value = nullptr (LRTAstar doesn't use
                    // terminal_value)
           false, time_budget, rollout_budget, max_depth, 100, 0.0, 1.0, false,
           callback, verbose) {}

SK_LRTASTAR_SOLVER_TEMPLATE_DECL
SK_LRTASTAR_SOLVER_CLASS::LRTAstarSolver(
    Tdomain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic,
    const typename Base::TerminalValueFunctor &, bool, std::size_t time_budget,
    std::size_t rollout_budget, std::size_t max_depth, std::size_t, double,
    double, bool, const CallbackFunctor &callback, bool verbose)
    : Base(domain, goal_checker, heuristic,
           nullptr, // terminal_value = nullptr (LRTAstar doesn't use
                    // terminal_value)
           false, time_budget, rollout_budget, max_depth, 100, 0.0, 1.0, false,
           callback, verbose) {}

SK_LRTASTAR_SOLVER_TEMPLATE_DECL
std::vector<typename Tdomain::Action>
SK_LRTASTAR_SOLVER_CLASS::get_plan(const State &s) const {
  std::vector<Action> plan;
  auto si = this->_graph.find(s);
  while (si != this->_graph.end() && si->best_action != nullptr) {
    plan.push_back(si->best_action->action);
    if (si->best_action->outcomes.empty()) {
      break;
    }
    auto *next = std::get<2>(si->best_action->outcomes.front());
    if (next->goal) {
      break;
    }
    si = this->_graph.find(next->state);
  }
  return plan;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_LRTDP_SOLVER_CLASS>
SK_LRTDP_SOLVER_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> goal_checker,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> terminal_value, const Params &params,
    bool verbose) {
  auto wrapped_gc = [goal_checker](Domain &d, const State &s,
                                   const std::size_t *) {
    return goal_checker(d, s);
  };
  auto wrapped_h = [heuristic](Domain &d, const State &s, const std::size_t *) {
    return heuristic(d, s);
  };
  return std::make_unique<LRTDPSolver>(
      domain, wrapped_gc, wrapped_h, terminal_value,
      params.template get<bool>("use_labels", true),
      params.template get<std::size_t>("time_budget", 3600000),
      params.template get<std::size_t>("rollout_budget", 100000),
      params.template get<std::size_t>("max_depth", 1000),
      params.template get<std::size_t>("residual_moving_average_window", 100),
      params.template get<double>("epsilon", 0.001),
      params.template get<double>("discount", 1.0),
      params.template get<bool>("online_node_garbage", false),
      CallbackFunctor([](const LRTDPSolver &, Domain &, const std::size_t *) {
        return false;
      }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_LRTDP_IMPL_HH
