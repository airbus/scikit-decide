/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_AOSTAR_IMPL_HH
#define SKDECIDE_AOSTAR_IMPL_HH

#include <chrono>

#include "utils/string_converter.hh"
#include "utils/logging.hh"

namespace skdecide {

// === AOStarSolver implementation ===

#define SK_AOSTAR_SOLVER_TEMPLATE_DECL                                         \
  template <typename Tdomain, typename Texecution_policy>

#define SK_AOSTAR_SOLVER_CLASS AOStarSolver<Tdomain, Texecution_policy>

SK_AOSTAR_SOLVER_TEMPLATE_DECL
SK_AOSTAR_SOLVER_CLASS::AOStarSolver(
    Domain &domain, const GoalCheckerFunctor &goal_checker,
    const HeuristicFunctor &heuristic, double discount,
    std::size_t max_tip_expansions, bool detect_cycles,
    const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _discount(discount), _max_tip_expansions(max_tip_expansions),
      _detect_cycles(detect_cycles), _callback(callback), _verbose(verbose) {

  if (verbose) {
    Logger::check_level(logging::debug, "algorithm AO*");
  }
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
void SK_AOSTAR_SOLVER_CLASS::clear() {
  _priority_queue = PriorityQueue();
  _graph.clear();
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
void SK_AOSTAR_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " AO* solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    auto si = _graph.emplace(s);
    if (si.first->solved ||
        _goal_checker(_domain,
                      s)) { // problem already solved from this state (was
                            // present in _graph and already solved)
      return;
    }
    StateNode &root_node = const_cast<StateNode &>(
        *(si.first)); // we won't change the real key (StateNode::state) so we
                      // are safe
    _priority_queue =
        PriorityQueue(); // contains only non-goal unsolved tip nodes
    _priority_queue.push(&root_node);

    while (!_priority_queue.empty() && !_callback(*this, _domain)) {
      if (_verbose) {
        Logger::debug("Current number of tip nodes: " +
                      StringConverter::from(_priority_queue.size()));
        Logger::debug("Current number of explored nodes: " +
                      StringConverter::from(_graph.size()));
      }
      std::size_t nb_expansions =
          std::min(_priority_queue.size(), _max_tip_expansions);
      std::unordered_set<StateNode *> frontier;
      for (std::size_t cnt = 0; cnt < nb_expansions; cnt++) {
        // Select best tip node of best partial graph
        StateNode *best_tip_node = _priority_queue.top();
        _priority_queue.pop();
        frontier.insert(best_tip_node);
        if (_verbose)
          Logger::debug("Current best tip node: " +
                        best_tip_node->state.print());

        // Expand best tip node
        auto applicable_actions =
            _domain.get_applicable_actions(best_tip_node->state).get_elements();
        std::for_each(
            ExecutionPolicy::policy, applicable_actions.begin(),
            applicable_actions.end(), [this, &best_tip_node](auto a) {
              if (_verbose)
                Logger::debug("Current expanded action: " + a.print() +
                              ExecutionPolicy::print_thread());
              _execution_policy.protect([&best_tip_node, &a] {
                best_tip_node->actions.push_back(
                    std::make_unique<ActionNode>(a));
              });
              ActionNode &an = *(best_tip_node->actions.back());
              an.parent = best_tip_node;
              auto next_states =
                  _domain.get_next_state_distribution(best_tip_node->state, a)
                      .get_values();
              for (auto ns : next_states) {
                if (_verbose)
                  Logger::debug(
                      "Current next state expansion: " + ns.state().print() +
                      ExecutionPolicy::print_thread());
                std::pair<typename Graph::iterator, bool> i;
                _execution_policy.protect(
                    [this, &i, &ns] { i = _graph.emplace(ns.state()); });
                StateNode &next_node = const_cast<StateNode &>(
                    *(i.first)); // we won't change the real key
                                 // (StateNode::state) so we are safe
                an.outcomes.push_back(std::make_tuple(
                    ns.probability(),
                    _domain
                        .get_transition_value(best_tip_node->state, a,
                                              next_node.state)
                        .cost(),
                    &next_node));
                _execution_policy.protect(
                    [&next_node, &an] { next_node.parents.push_back(&an); });
                if (i.second) { // new node
                  if (_goal_checker(_domain, next_node.state)) {
                    if (_verbose)
                      Logger::debug("Found goal state " +
                                    next_node.state.print() +
                                    ExecutionPolicy::print_thread());
                    next_node.solved = true;
                    next_node.best_value = 0.0;
                  } else {
                    next_node.best_value =
                        _heuristic(_domain, next_node.state).cost();
                    if (_verbose)
                      Logger::debug(
                          "New state " + next_node.state.print() +
                          " with heuristic value " +
                          StringConverter::from(next_node.best_value) +
                          ExecutionPolicy::print_thread());
                  }
                }
              }
            });
      }

      // Back-propagate value function from best tip node
      std::unique_ptr<std::unordered_set<StateNode *>>
          explored_states; // only for detecting cycles
      if (_detect_cycles)
        explored_states =
            std::make_unique<std::unordered_set<StateNode *>>(frontier);
      while (!frontier.empty()) {
        std::unordered_set<StateNode *> new_frontier;
        std::for_each(ExecutionPolicy::policy, frontier.begin(), frontier.end(),
                      [this, &new_frontier](const auto &fs) {
                        // update Q-values and V-value
                        fs->best_value =
                            std::numeric_limits<double>::infinity();
                        fs->best_action = nullptr;
                        for (const auto &a : fs->actions) {
                          a->value = 0.0;
                          for (const auto &ns : a->outcomes) {
                            a->value +=
                                std::get<0>(ns) *
                                (std::get<1>(ns) +
                                 (_discount * std::get<2>(ns)->best_value));
                          }
                          if (a->value < fs->best_value) {
                            fs->best_value = a->value;
                            fs->best_action = a.get();
                          }
                          fs->best_value = std::min(fs->best_value, a->value);
                        }
                        // update solved field
                        fs->solved = true;
                        for (const auto &ns : fs->best_action->outcomes) {
                          fs->solved = fs->solved && std::get<2>(ns)->solved;
                        }
                        // update new frontier
                        _execution_policy.protect([&fs, &new_frontier] {
                          for (const auto &ps : fs->parents) {
                            new_frontier.insert(ps->parent);
                          }
                        });
                      });
        frontier = new_frontier;
        if (_detect_cycles) {
          for (const auto &ps : new_frontier) {
            if (explored_states->find(ps) != explored_states->end()) {
              throw std::logic_error("SKDECIDE exception: cycle detected in "
                                     "the MDP graph! [with state " +
                                     ps->state.print() + "]");
            }
            explored_states->insert(ps);
          }
        }
      }

      // Recompute best partial graph
      _priority_queue = PriorityQueue();
      frontier.insert(&root_node);
      while (!frontier.empty()) {
        std::unordered_set<StateNode *> new_frontier;
        for (const auto &fs : frontier) {
          if (!(fs->solved)) {
            if (fs->best_action != nullptr) {
              for (const auto &ns : fs->best_action->outcomes) {
                new_frontier.insert(std::get<2>(ns));
              }
            } else { // tip node
              _priority_queue.push(fs);
            }
          }
        }
        frontier = new_frontier;
      }
    }

    Logger::info(
        "AO* finished to solve from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds.");
  } catch (const std::exception &e) {
    Logger::error("AO* failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
bool SK_AOSTAR_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr) ||
      (si->solved == false)) {
    return false;
  } else {
    return true;
  }
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
const typename SK_AOSTAR_SOLVER_CLASS::Action &
SK_AOSTAR_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return si->best_action->action;
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
typename SK_AOSTAR_SOLVER_CLASS::Value
SK_AOSTAR_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  Value val;
  val.cost(si->best_value);
  return val;
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
std::size_t SK_AOSTAR_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_AOSTAR_SOLVER_CLASS::State>::Set
SK_AOSTAR_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored_states;
  for (const auto &s : _graph) {
    explored_states.insert(s.state);
  }
  return explored_states;
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL std::size_t
SK_AOSTAR_SOLVER_CLASS::get_nb_tip_states() const {
  return _priority_queue.size();
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
const typename SK_AOSTAR_SOLVER_CLASS::State &
SK_AOSTAR_SOLVER_CLASS::get_top_tip_state() const {
  if (_priority_queue.empty()) {
    Logger::error(
        "SKDECIDE exception: no top tip state (empty priority queue)");
    throw std::runtime_error(
        "SKDECIDE exception: no top tip state (empty priority queue)");
  }
  return _priority_queue.top()->state;
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
std::size_t SK_AOSTAR_SOLVER_CLASS::get_solving_time() const {
  std::size_t milliseconds_duration;
  milliseconds_duration = static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
  return milliseconds_duration;
}

SK_AOSTAR_SOLVER_TEMPLATE_DECL typename MapTypeDeducer<
    typename SK_AOSTAR_SOLVER_CLASS::State,
    std::pair<typename SK_AOSTAR_SOLVER_CLASS::Action,
              typename SK_AOSTAR_SOLVER_CLASS::Value>>::Map
SK_AOSTAR_SOLVER_CLASS::get_policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (auto &n : _graph) {
    if (n.best_action != nullptr) {
      Value val;
      val.cost(n.best_value);
      p.insert(
          std::make_pair(n.state, std::make_pair(n.best_action->action, val)));
    }
  }
  return p;
}

// === AOStarSolver::StateNode implementation ===

SK_AOSTAR_SOLVER_TEMPLATE_DECL
SK_AOSTAR_SOLVER_CLASS::StateNode::StateNode(const State &s)
    : state(s), best_action(nullptr),
      best_value(std::numeric_limits<double>::infinity()), solved(false) {}

SK_AOSTAR_SOLVER_TEMPLATE_DECL
const typename SK_AOSTAR_SOLVER_CLASS::State &
SK_AOSTAR_SOLVER_CLASS::StateNode::Key::operator()(const StateNode &sn) const {
  return sn.state;
}

// === AOStarSolver::ActionNode implementation ===

SK_AOSTAR_SOLVER_TEMPLATE_DECL
SK_AOSTAR_SOLVER_CLASS::ActionNode::ActionNode(const Action &a)
    : action(a), value(std::numeric_limits<double>::infinity()),
      parent(nullptr) {}

// === AOStarSolver::StateNodeCompare implementation ===

SK_AOSTAR_SOLVER_TEMPLATE_DECL
bool SK_AOSTAR_SOLVER_CLASS::StateNodeCompare::operator()(StateNode *&a,
                                                          StateNode *&b) const {
  return (a->best_value) >
         (b->best_value); // smallest element appears at the top of the
                          // priority_queue => cost optimization
}

} // namespace skdecide

#endif // SKDECIDE_AOSTAR_IMPL_HH
