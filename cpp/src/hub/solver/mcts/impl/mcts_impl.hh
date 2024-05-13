/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_IMPL_HH
#define SKDECIDE_MCTS_IMPL_HH

#include <boost/range/irange.hpp>
#include <iostream>

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === MCTSSolver implementation ===

#define SK_MCTS_SOLVER_TEMPLATE_DECL                                           \
  template <typename Tdomain, typename TexecutionPolicy,                       \
            template <typename Tsolver> class TtransitionMode,                 \
            template <typename Tsolver> class TtreePolicy,                     \
            template <typename Tsolver> class Texpander,                       \
            template <typename Tsolver> class TactionSelectorOptimization,     \
            template <typename Tsolver> class TactionSelectorExecution,        \
            template <typename Tsolver> class TrolloutPolicy,                  \
            template <typename Tsolver> class TbackPropagator>

#define SK_MCTS_SOLVER_CLASS                                                   \
  MCTSSolver<Tdomain, TexecutionPolicy, TtransitionMode, TtreePolicy,          \
             Texpander, TactionSelectorOptimization, TactionSelectorExecution, \
             TrolloutPolicy, TbackPropagator>

SK_MCTS_SOLVER_TEMPLATE_DECL
SK_MCTS_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), expansions_count(0), value(0.0), visits_count(0),
      parent(nullptr) {}

SK_MCTS_SOLVER_TEMPLATE_DECL
SK_MCTS_SOLVER_CLASS::ActionNode::ActionNode(const ActionNode &a)
    : action(a.action), outcomes(a.outcomes),
      dist_to_outcome(a.dist_to_outcome), dist(a.dist),
      expansions_count((std::size_t)a.expansions_count), value((double)a.value),
      visits_count((std::size_t)a.visits_count), parent(a.parent) {}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::Action &
SK_MCTS_SOLVER_CLASS::ActionNode::Key::operator()(const ActionNode &an) const {
  return an.action;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
SK_MCTS_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), terminal(false), expanded(false), expansions_count(0),
      value(0.0), visits_count(0) {}

SK_MCTS_SOLVER_TEMPLATE_DECL
SK_MCTS_SOLVER_CLASS::StateNode::StateNode(const StateNode &s)
    : state(s.state), terminal((bool)s.terminal), expanded((bool)s.expanded),
      expansions_count((std::size_t)s.expansions_count), actions(s.actions),
      value((double)s.value), visits_count((std::size_t)s.visits_count),
      parents(s.parents) {}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::State &
SK_MCTS_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
SK_MCTS_SOLVER_CLASS::MCTSSolver(
    Domain &domain, std::size_t time_budget, std::size_t rollout_budget,
    std::size_t max_depth, std::size_t residual_moving_average_window,
    double epsilon, double discount, bool online_node_garbage, bool debug_logs,
    const CallbackFunctor &callback, std::unique_ptr<TreePolicy> tree_policy,
    std::unique_ptr<Expander> expander,
    std::unique_ptr<ActionSelectorOptimization> action_selector_optimization,
    std::unique_ptr<ActionSelectorExecution> action_selector_execution,
    std::unique_ptr<RolloutPolicy> rollout_policy,
    std::unique_ptr<BackPropagator> back_propagator)
    : _domain(domain), _time_budget(time_budget),
      _rollout_budget(rollout_budget), _max_depth(max_depth),
      _residual_moving_average_window(residual_moving_average_window),
      _epsilon(epsilon), _discount(discount), _nb_rollouts(0),
      _online_node_garbage(online_node_garbage), _debug_logs(debug_logs),
      _callback(callback),
      _execution_policy(std::make_unique<ExecutionPolicy>()),
      _transition_mode(std::make_unique<TransitionMode>()),
      _tree_policy(std::move(tree_policy)), _expander(std::move(expander)),
      _action_selector_optimization(std::move(action_selector_optimization)),
      _action_selector_execution(std::move(action_selector_execution)),
      _rollout_policy(std::move(rollout_policy)),
      _back_propagator(std::move(back_propagator)), _current_state(nullptr),
      _residual_moving_average(0) {

  if (debug_logs) {
    Logger::check_level(logging::debug, "algorithm MCTS");
  }

  std::random_device rd;
  _gen = std::make_unique<std::mt19937>(rd());
}

SK_MCTS_SOLVER_TEMPLATE_DECL
void SK_MCTS_SOLVER_CLASS::clear() { _graph.clear(); }

SK_MCTS_SOLVER_TEMPLATE_DECL
void SK_MCTS_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " MCTS solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();
    _nb_rollouts = 0;
    _residual_moving_average = 0.0;
    _residuals.clear();

    // Get the root node
    auto si = _graph.emplace(s);
    StateNode &root_node = const_cast<StateNode &>(
        *(si.first)); // we won't change the real key (StateNode::state) so we
                      // are safe

    boost::integer_range<std::size_t> parallel_rollouts(
        0, _domain.get_parallel_capacity());

    std::for_each(
        ExecutionPolicy::policy, parallel_rollouts.begin(),
        parallel_rollouts.end(),
        [this, &root_node](const std::size_t &thread_id) {
          do {
            std::size_t depth = 0;
            double root_node_record_value = root_node.value;
            StateNode *sn = (*_tree_policy)(*this, &thread_id, *_expander,
                                            *_action_selector_optimization,
                                            root_node, depth);
            (*_rollout_policy)(*this, &thread_id, *sn, depth);
            (*_back_propagator)(*this, &thread_id, *sn);
            update_residual_moving_average(root_node, root_node_record_value);
            _nb_rollouts++;
          } while (!_callback(*this, _domain, &thread_id) &&
                   (get_solving_time() < _time_budget) &&
                   (_nb_rollouts < _rollout_budget) &&
                   (get_residual_moving_average() > _epsilon));
        });

    Logger::info(
        "MCTS finished to solve from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e6) +
        " seconds with " + StringConverter::from(_nb_rollouts) + " rollouts.");
  } catch (const std::exception &e) {
    Logger::error("MCTS failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
bool SK_MCTS_SOLVER_CLASS::is_solution_defined_for(const State &s) {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    return false;
  } else {
    return (*_action_selector_execution)(*this, nullptr, *si) != nullptr;
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
typename SK_MCTS_SOLVER_CLASS::Action
SK_MCTS_SOLVER_CLASS::get_best_action(const State &s) {
  auto si = _graph.find(s);
  ActionNode *action = nullptr;
  if (si != _graph.end()) {
    action = (*_action_selector_execution)(*this, nullptr, *si);
  }
  if (action == nullptr) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  } else {
    if (_debug_logs) {
      std::string str = "(";
      for (const auto &o : action->outcomes) {
        str += "\n    " + o.first->state.print();
      }
      str += "\n)";
      Logger::debug("Best action's known outcomes:\n" + str);
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
    _action_prefix.push_back(action->action);
    return action->action;
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
typename SK_MCTS_SOLVER_CLASS::Value
SK_MCTS_SOLVER_CLASS::get_best_value(const State &s) {
  auto si = _graph.find(StateNode(s));
  ActionNode *action = nullptr;
  if (si != _graph.end()) {
    action = (*_action_selector_execution)(*this, nullptr, *si);
  }
  if (action == nullptr) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  } else {
    Value val;
    val.reward(action->value);
    return val;
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
std::size_t SK_MCTS_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_MCTS_SOLVER_TEMPLATE_DECL
std::size_t SK_MCTS_SOLVER_CLASS::get_nb_rollouts() const {
  return _nb_rollouts;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
double SK_MCTS_SOLVER_CLASS::get_residual_moving_average() const {
  if (_residuals.size() >= _residual_moving_average_window) {
    return (double)_residual_moving_average;
  } else {
    return std::numeric_limits<double>::infinity();
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
std::size_t SK_MCTS_SOLVER_CLASS::get_solving_time() {
  std::size_t milliseconds_duration;
  _execution_policy->protect(
      [this, &milliseconds_duration]() {
        milliseconds_duration = static_cast<std::size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - _start_time)
                .count());
      },
      _time_mutex);
  return milliseconds_duration;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<typename SK_MCTS_SOLVER_CLASS::State,
                        std::pair<typename SK_MCTS_SOLVER_CLASS::Action,
                                  typename SK_MCTS_SOLVER_CLASS::Value>>::Map
SK_MCTS_SOLVER_CLASS::get_policy() {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (auto &n : _graph) {
    ActionNode *action = (*_action_selector_execution)(*this, nullptr, n);
    if (action != nullptr) {
      Value val;
      val.reward(action->value);
      p.insert(std::make_pair(n.state, std::make_pair(action->action, val)));
    }
  }
  return p;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
typename SK_MCTS_SOLVER_CLASS::Domain &SK_MCTS_SOLVER_CLASS::domain() {
  return _domain;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
std::size_t SK_MCTS_SOLVER_CLASS::time_budget() const { return _time_budget; }

SK_MCTS_SOLVER_TEMPLATE_DECL
std::size_t SK_MCTS_SOLVER_CLASS::rollout_budget() const {
  return _rollout_budget;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
std::size_t SK_MCTS_SOLVER_CLASS::max_depth() const { return _max_depth; }

SK_MCTS_SOLVER_TEMPLATE_DECL
double SK_MCTS_SOLVER_CLASS::discount() const { return _discount; }

SK_MCTS_SOLVER_TEMPLATE_DECL
typename SK_MCTS_SOLVER_CLASS::ExecutionPolicy &
SK_MCTS_SOLVER_CLASS::execution_policy() {
  return *_execution_policy;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::TransitionMode &
SK_MCTS_SOLVER_CLASS::transition_mode() {
  return *_transition_mode;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::TreePolicy &
SK_MCTS_SOLVER_CLASS::tree_policy() {
  return *_tree_policy;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::Expander &
SK_MCTS_SOLVER_CLASS::expander() {
  return *_expander;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::ActionSelectorOptimization &
SK_MCTS_SOLVER_CLASS::action_selector_optimization() {
  return *_action_selector_optimization;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::ActionSelectorExecution &
SK_MCTS_SOLVER_CLASS::action_selector_execution() {
  return *_action_selector_execution;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::RolloutPolicy &
SK_MCTS_SOLVER_CLASS::rollout_policy() {
  return *_rollout_policy;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const typename SK_MCTS_SOLVER_CLASS::BackPropagator &
SK_MCTS_SOLVER_CLASS::back_propagator() {
  return *_back_propagator;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
typename SK_MCTS_SOLVER_CLASS::Graph &SK_MCTS_SOLVER_CLASS::graph() {
  return _graph;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
const std::list<typename SK_MCTS_SOLVER_CLASS::Action> &
SK_MCTS_SOLVER_CLASS::action_prefix() const {
  return _action_prefix;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
std::mt19937 &SK_MCTS_SOLVER_CLASS::gen() { return *_gen; }

SK_MCTS_SOLVER_TEMPLATE_DECL
typename SK_MCTS_SOLVER_CLASS::ExecutionPolicy::Mutex &
SK_MCTS_SOLVER_CLASS::gen_mutex() {
  return _gen_mutex;
}

SK_MCTS_SOLVER_TEMPLATE_DECL
bool SK_MCTS_SOLVER_CLASS::debug_logs() const { return _debug_logs; }

SK_MCTS_SOLVER_TEMPLATE_DECL
void SK_MCTS_SOLVER_CLASS::compute_reachable_subgraph(
    StateNode *node, std::unordered_set<StateNode *> &subgraph) {
  std::unordered_set<StateNode *> frontier;
  frontier.insert(node);
  subgraph.insert(node);
  while (!frontier.empty()) {
    std::unordered_set<StateNode *> new_frontier;
    for (auto &n : frontier) {
      for (auto &action : n->actions) {
        for (auto &outcome : action.outcomes) {
          if (subgraph.find(outcome.first) == subgraph.end()) {
            new_frontier.insert(outcome.first);
            subgraph.insert(outcome.first);
          }
        }
      }
    }
    frontier = new_frontier;
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
void SK_MCTS_SOLVER_CLASS::remove_subgraph(
    std::unordered_set<StateNode *> &root_subgraph,
    std::unordered_set<StateNode *> &child_subgraph) {
  std::unordered_set<StateNode *> removed_subgraph;
  // First pass: look for nodes in root_subgraph but not child_subgraph and
  // remove those nodes from their children's parents Don't actually remove
  // those nodes in the first pass otherwise some children to remove won't exist
  // anymore when looking for their parents
  for (auto &n : root_subgraph) {
    if (child_subgraph.find(n) == child_subgraph.end()) {
      for (auto &action : n->actions) {
        for (auto &outcome : action.outcomes) {
          // we won't change the real key (ActionNode::action) so we are safe
          outcome.first->parents.erase(&const_cast<ActionNode &>(action));
        }
      }
      removed_subgraph.insert(n);
    }
  }
  // Second pass: actually remove nodes in root_subgraph but not in
  // child_subgraph
  for (auto &n : removed_subgraph) {
    _graph.erase(StateNode(n->state));
  }
}

SK_MCTS_SOLVER_TEMPLATE_DECL
void SK_MCTS_SOLVER_CLASS::update_residual_moving_average(
    const StateNode &node, const double &node_record_value) {
  if (_residual_moving_average_window > 0) {
    double current_epsilon = std::fabs(node_record_value - node.value);
    _execution_policy->protect(
        [this, &current_epsilon]() {
          if (_residuals.size() < _residual_moving_average_window) {
            _residual_moving_average =
                ((double)((_residual_moving_average * _residuals.size()) +
                          current_epsilon)) /
                ((double)(_residuals.size() + 1));
          } else {
            _residual_moving_average =
                ((double)_residual_moving_average) +
                ((current_epsilon - _residuals.front()) /
                 ((double)_residual_moving_average_window));
            _residuals.pop_front();
          }
          _residuals.push_back(current_epsilon);
        },
        _residuals_protect);
  }
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_IMPL_HH
