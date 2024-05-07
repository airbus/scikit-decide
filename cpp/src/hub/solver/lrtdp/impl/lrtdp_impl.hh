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
    const HeuristicFunctor &heuristic, bool use_labels, std::size_t time_budget,
    std::size_t rollout_budget, std::size_t max_depth,
    std::size_t residual_moving_average_window, double epsilon, double discount,
    bool online_node_garbage, bool debug_logs, const CallbackFunctor &callback)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _use_labels(use_labels), _time_budget(time_budget),
      _rollout_budget(rollout_budget), _max_depth(max_depth),
      _residual_moving_average_window(residual_moving_average_window),
      _epsilon(epsilon), _discount(discount),
      _online_node_garbage(online_node_garbage), _debug_logs(debug_logs),
      _callback(callback), _current_state(nullptr), _nb_rollouts(0) {

  if (debug_logs) {
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
                      if (_debug_logs)
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
        StringConverter::from((double)get_solving_time() / (double)1e6) +
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
bool SK_LRTDP_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    // /!\ does not mean the state is solved!
    return false;
  } else {
    return true;
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
const typename SK_LRTDP_SOLVER_CLASS::Action &
SK_LRTDP_SOLVER_CLASS::get_best_action(const State &s) {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  } else {
    if (_debug_logs) {
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
          child_subgraph); // we won't change the real key (StateNode::state) so
                           // we are safe
      remove_subgraph(root_subgraph, child_subgraph);
    }
    _current_state = const_cast<StateNode *>(&(
        *si)); // we won't change the real key (StateNode::state) so we are safe
    return si->best_action->action;
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
double SK_LRTDP_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_value;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_LRTDP_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
std::size_t SK_LRTDP_SOLVER_CLASS::get_nb_rollouts() const {
  return _nb_rollouts;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
double SK_LRTDP_SOLVER_CLASS::get_residual_moving_average() const {
  if (_residuals.size() >= _residual_moving_average_window) {
    return (double)_residual_moving_average;
  } else {
    return std::numeric_limits<double>::infinity();
  }
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
    std::pair<typename SK_LRTDP_SOLVER_CLASS::Action, double>>::Map
SK_LRTDP_SOLVER_CLASS::get_policy() const {
  typename MapTypeDeducer<State, std::pair<Action, double>>::Map p;
  for (auto &n : _graph) {
    if (n.best_action != nullptr) {
      p.insert(std::make_pair(n.state, std::make_pair(n.best_action->action,
                                                      (double)n.best_value)));
    }
  }
  return p;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::expand(StateNode *s, const std::size_t *thread_id) {
  if (_debug_logs)
    Logger::debug("Expanding state " + s->state.print() +
                  ExecutionPolicy::print_thread());
  auto applicable_actions =
      _domain.get_applicable_actions(s->state, thread_id).get_elements();

  for (auto a : applicable_actions) {
    if (_debug_logs)
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
          [this, &i, &ns] { i = _graph.emplace(ns.state()); });
      StateNode &next_node = const_cast<StateNode &>(
          *(i.first)); // we won't change the real key (StateNode::state) so
                       // we are safe
      an.outcomes.push_back(std::make_tuple(
          ns.probability(),
          _domain.get_transition_value(s->state, a, next_node.state, thread_id)
              .cost(),
          &next_node));
      outcome_weights.push_back(std::get<0>(an.outcomes.back()));
      if (_debug_logs)
        Logger::debug(
            "Current next state expansion: " + next_node.state.print() +
            ExecutionPolicy::print_thread());

      if (i.second) { // new node
        if (_goal_checker(_domain, next_node.state, thread_id)) {
          if (_debug_logs)
            Logger::debug("Found goal state " + next_node.state.print() +
                          ExecutionPolicy::print_thread());
          next_node.goal = true;
          next_node.solved = true;
          next_node.best_value = 0.0;
        } else {
          next_node.best_value =
              _heuristic(_domain, next_node.state, thread_id).cost();
          if (_debug_logs)
            Logger::debug("New state " + next_node.state.print() +
                          " with heuristic value " +
                          StringConverter::from(next_node.best_value) +
                          ExecutionPolicy::print_thread());
        }
      }
    }

    an.dist = std::discrete_distribution<>(outcome_weights.begin(),
                                           outcome_weights.end());
  }
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
double SK_LRTDP_SOLVER_CLASS::q_value(ActionNode *a) {
  a->value = 0;
  for (const auto &o : a->outcomes) {
    a->value = a->value +
               (std::get<0>(o) *
                (std::get<1>(o) + (_discount * std::get<2>(o)->best_value)));
  }
  if (_debug_logs)
    Logger::debug("Updated Q-value of action " + a->action.print() +
                  " with value " + StringConverter::from(a->value) +
                  ExecutionPolicy::print_thread());
  return a->value;
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
      best_value = a->value;
      best_action = a.get();
    }
  }

  if (_debug_logs) {
    Logger::debug("Greedy action of state " + s->state.print() + ": " +
                  best_action->action.print() + " with value " +
                  StringConverter::from(best_value) +
                  ExecutionPolicy::print_thread());
  }

  return best_action;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
void SK_LRTDP_SOLVER_CLASS::update(StateNode *s, const std::size_t *thread_id) {
  if (_debug_logs)
    Logger::debug("Updating state " + s->state.print() +
                  ExecutionPolicy::print_thread());
  s->best_action = greedy_action(s, thread_id);
  s->best_value = (double)s->best_action->value;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
typename SK_LRTDP_SOLVER_CLASS::StateNode *
SK_LRTDP_SOLVER_CLASS::pick_next_state(ActionNode *a) {
  StateNode *s = nullptr;
  _execution_policy.protect(
      [&a, &s, this]() {
        s = std::get<2>(a->outcomes[a->dist(*_gen)]);
        if (_debug_logs)
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
  double res = std::fabs(s->best_value - s->best_action->value);
  if (_debug_logs)
    Logger::debug("State " + s->state.print() + " has residual " +
                  StringConverter::from(res) + ExecutionPolicy::print_thread());
  return res;
}

SK_LRTDP_SOLVER_TEMPLATE_DECL
bool SK_LRTDP_SOLVER_CLASS::check_solved(StateNode *s,
                                         const std::size_t *thread_id) {
  if (_debug_logs) {
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
  std::unordered_set<StateNode *> visited;
  std::size_t depth = 0;

  if (!(s->solved)) {
    open.push(s);
    visited.insert(s);
  }

  while (!open.empty() && (get_solving_time() < _time_budget) &&
         (depth < _max_depth)) {
    depth++;
    StateNode *cs = open.top();
    open.pop();
    closed.push(cs);
    visited.insert(cs);

    _execution_policy.protect(
        [this, &cs, &rv, &open, &visited, &thread_id]() {
          if (residual(cs, thread_id) > _epsilon) {
            rv = false;
            return;
          }

          ActionNode *a = cs->best_action; // best action updated when calling
                                           // residual(cs, thread_id)
          for (const auto &o : a->outcomes) {
            StateNode *ns = std::get<2>(o);
            if (!(ns->solved) && (visited.find(ns) == visited.end())) {
              open.push(ns);
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

  if (_debug_logs) {
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
  StateNode *cs = s;
  std::size_t depth = 0;
  bool found_goal = false;

  while ((!_use_labels || !(cs->solved)) && !found_goal &&
         (get_solving_time() < _time_budget) && (depth < _max_depth)) {
    depth++;
    visited.push(cs);
    _execution_policy.protect(
        [this, &cs, &found_goal, &thread_id]() {
          if (cs->goal) {
            if (_debug_logs)
              Logger::debug("Found goal state " + cs->state.print() +
                            ExecutionPolicy::print_thread());
            found_goal = true;
          }

          update(cs, thread_id);
          cs = pick_next_state(cs->best_action);
        },
        cs->mutex);
  }

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
    double current_residual = std::fabs(node_record_value - node.best_value);
    _execution_policy.protect(
        [this, &current_residual]() {
          if (_residuals.size() < _residual_moving_average_window) {
            _residual_moving_average =
                ((double)((_residual_moving_average * _residuals.size()) +
                          current_residual)) /
                ((double)(_residuals.size() + 1));
          } else {
            _residual_moving_average =
                ((double)_residual_moving_average) +
                ((current_residual - _residuals.front()) /
                 ((double)_residual_moving_average_window));
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

} // namespace skdecide

#endif // SKDECIDE_LRTDP_IMPL_HH
