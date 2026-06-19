/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_EHC_IMPL_HH
#define SKDECIDE_EHC_IMPL_HH

#include <chrono>
#include <queue>
#include <stdexcept>
#include <unordered_set>

#include "utils/logging.hh"
#include "utils/string_converter.hh"

namespace skdecide {

#define SK_EHC_SOLVER_TEMPLATE_DECL                                            \
  template <typename Tdomain, typename Texecution_policy>

#define SK_EHC_SOLVER_CLASS EHCSolver<Tdomain, Texecution_policy>

SK_EHC_SOLVER_TEMPLATE_DECL
SK_EHC_SOLVER_CLASS::EHCSolver(Domain &domain,
                               const GoalCheckerFunctor &goal_checker,
                               const HeuristicFunctor &heuristic,
                               const PreferredActionsFunctor &preferred_actions,
                               const CallbackFunctor &callback, bool verbose)
    : _domain(domain), _goal_checker(goal_checker), _heuristic(heuristic),
      _preferred_actions(preferred_actions), _callback(callback),
      _verbose(verbose) {
  if (verbose) {
    Logger::check_level(logging::debug, "algorithm EHC");
  }
}

SK_EHC_SOLVER_TEMPLATE_DECL
void SK_EHC_SOLVER_CLASS::clear() { _graph.clear(); }

SK_EHC_SOLVER_TEMPLATE_DECL
void SK_EHC_SOLVER_CLASS::solve(const State &s) {
  try {
    Logger::info("Running " + ExecutionPolicy::print_type() +
                 " EHC solver from state " + s.print());
    _start_time = std::chrono::high_resolution_clock::now();

    auto si = _graph.emplace(s);
    Node *current = &const_cast<Node &>(*(si.first));

    if (_goal_checker(_domain, current->state)) {
      current->solved = true;
      Logger::info("EHC: initial state is already a goal.");
      return;
    }

    double h_current = _heuristic(_domain, current->state).cost();

    if (_verbose)
      Logger::debug("EHC initial h-value: " + StringConverter::from(h_current));

    while (!_goal_checker(_domain, current->state)) {
      if (_callback(*this, _domain)) {
        Logger::info("EHC interrupted by callback.");
        return;
      }

      // BFS for an improving state
      std::queue<Node *> bfs_queue;
      std::unordered_set<Node *> bfs_visited;
      bfs_queue.push(current);
      bfs_visited.insert(current);

      bool found_improving = false;
      Node *improving_node = nullptr;

      while (!bfs_queue.empty() && !found_improving) {
        Node *node = bfs_queue.front();
        bfs_queue.pop();

        if (_callback(*this, _domain)) {
          Logger::info("EHC interrupted by callback.");
          return;
        }

        // Get applicable actions (sequential — may not be thread-safe)
        auto applicable_actions =
            _domain.get_applicable_actions(node->state).get_elements();

        // If preferred actions functor is set, partition into preferred first
        std::vector<Action> preferred;
        std::unordered_set<std::size_t> preferred_hashes;

        if (_preferred_actions) {
          preferred = _preferred_actions(_domain, node->state);
          for (auto &pa : preferred) {
            preferred_hashes.insert(typename Action::Hash()(pa));
          }
        }

        // Lambda to expand a batch of actions with parallel successor
        // generation
        auto expand_actions = [&](auto &actions) {
          std::for_each(
              ExecutionPolicy::policy, actions.begin(), actions.end(),
              [&](auto a) {
                if (found_improving)
                  return;

                auto next_state = _domain.get_next_state(node->state, a);
                auto transition_value =
                    _domain.get_transition_value(node->state, a, next_state);

                std::pair<typename Graph::iterator, bool> ins;
                _execution_policy.protect([this, &ins, &next_state] {
                  ins = _graph.emplace(next_state);
                });

                Node &neighbor = const_cast<Node &>(*(ins.first));

                if (ins.second) {
                  // New node — set parent
                  neighbor.best_parent =
                      std::make_tuple(node, a, transition_value.cost());

                  _execution_policy.protect([&] {
                    if (!bfs_visited.count(&neighbor)) {
                      bfs_visited.insert(&neighbor);
                      bfs_queue.push(&neighbor);
                    }
                  });
                }
              });
        };

        // Expand preferred actions first, then non-preferred
        if (!preferred.empty()) {
          expand_actions(preferred);

          if (!found_improving) {
            std::vector<Action> non_preferred;
            for (auto a : applicable_actions) {
              if (!preferred_hashes.count(typename Action::Hash()(a))) {
                non_preferred.push_back(a);
              }
            }
            expand_actions(non_preferred);
          }
        } else {
          expand_actions(applicable_actions);
        }

        // Evaluate newly queued nodes for improvement
        // (We must check nodes after expansion, not during, because
        //  heuristic evaluation should be sequential)
        std::size_t queue_size = bfs_queue.size();
        std::queue<Node *> requeue;

        for (std::size_t i = 0; i < queue_size && !found_improving; ++i) {
          Node *candidate = bfs_queue.front();
          bfs_queue.pop();

          if (_goal_checker(_domain, candidate->state)) {
            found_improving = true;
            improving_node = candidate;
            break;
          }

          double h_candidate = _heuristic(_domain, candidate->state).cost();

          if (_verbose)
            Logger::debug(
                "EHC BFS candidate h=" + StringConverter::from(h_candidate) +
                " (current=" + StringConverter::from(h_current) +
                ") state=" + candidate->state.print());

          if (h_candidate < h_current) {
            found_improving = true;
            improving_node = candidate;
          } else {
            requeue.push(candidate);
          }
        }

        // Put non-improving nodes back
        while (!requeue.empty()) {
          bfs_queue.push(requeue.front());
          requeue.pop();
        }
      }

      if (!found_improving) {
        Logger::error("EHC failed: no improving state found from state " +
                      current->state.print());
        throw std::runtime_error(
            "SKDECIDE exception: EHC failed — no improving state found");
      }

      // Backtrack path from improving_node to current and record policy
      std::vector<Node *> path;
      Node *trace = improving_node;
      while (trace != current) {
        path.push_back(trace);
        trace = std::get<0>(trace->best_parent);
      }

      // Record policy along the path (reverse order: from current to improving)
      for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        Node *child = path[i];
        Node *parent = std::get<0>(child->best_parent);
        parent->best_action =
            std::make_pair(&std::get<1>(child->best_parent), child);
        parent->solved = true;
      }

      h_current = _heuristic(_domain, improving_node->state).cost();
      current = improving_node;

      if (_verbose)
        Logger::debug("EHC improved to h=" + StringConverter::from(h_current) +
                      " state=" + current->state.print());
    }

    // Mark goal node as solved
    current->solved = true;

    Logger::info(
        "EHC finished solving from state " + s.print() + " in " +
        StringConverter::from((double)get_solving_time() / (double)1e3) +
        " seconds.");
  } catch (const std::exception &e) {
    Logger::error("EHC failed solving from state " + s.print() +
                  ". Reason: " + e.what());
    throw;
  }
}

SK_EHC_SOLVER_TEMPLATE_DECL
bool SK_EHC_SOLVER_CLASS::is_solution_defined_for(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action.first == nullptr &&
                               !_goal_checker(_domain, si->state))) {
    return false;
  }
  return true;
}

SK_EHC_SOLVER_TEMPLATE_DECL
const typename SK_EHC_SOLVER_CLASS::Action &
SK_EHC_SOLVER_CLASS::get_best_action(const State &s) const {
  auto si = _graph.find(s);
  if ((si == _graph.end()) || (si->best_action.first == nullptr)) {
    Logger::error("SKDECIDE exception: no best action found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best action found in state " + s.print());
  }
  return *(si->best_action.first);
}

SK_EHC_SOLVER_TEMPLATE_DECL
typename SK_EHC_SOLVER_CLASS::Value
SK_EHC_SOLVER_CLASS::get_best_value(const State &s) const {
  auto si = _graph.find(s);
  if (si == _graph.end()) {
    Logger::error("SKDECIDE exception: no best value found in state " +
                  s.print());
    throw std::runtime_error(
        "SKDECIDE exception: no best value found in state " + s.print());
  }
  Value val;
  if (si->best_action.second != nullptr) {
    val.cost(std::get<2>(si->best_action.second->best_parent));
  } else {
    val.cost(0.0);
  }
  return val;
}

SK_EHC_SOLVER_TEMPLATE_DECL
std::size_t SK_EHC_SOLVER_CLASS::get_nb_explored_states() const {
  return _graph.size();
}

SK_EHC_SOLVER_TEMPLATE_DECL
typename SetTypeDeducer<typename SK_EHC_SOLVER_CLASS::State>::Set
SK_EHC_SOLVER_CLASS::get_explored_states() const {
  typename SetTypeDeducer<State>::Set explored_states;
  for (const auto &n : _graph) {
    explored_states.insert(n.state);
  }
  return explored_states;
}

SK_EHC_SOLVER_TEMPLATE_DECL
std::size_t SK_EHC_SOLVER_CLASS::get_solving_time() const {
  return static_cast<std::size_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - _start_time)
          .count());
}

SK_EHC_SOLVER_TEMPLATE_DECL
std::vector<std::tuple<typename SK_EHC_SOLVER_CLASS::State,
                       typename SK_EHC_SOLVER_CLASS::Action,
                       typename SK_EHC_SOLVER_CLASS::Value>>
SK_EHC_SOLVER_CLASS::get_plan(const State &from_state) const {
  std::vector<std::tuple<State, Action, Value>> p;
  auto si = _graph.find(from_state);
  if (si == _graph.end()) {
    Logger::warn("SKDECIDE warning: no plan found starting in state " +
                 from_state.print());
    return p;
  }
  const Node *cur_node = &(*si);
  std::unordered_set<const Node *> plan_nodes;
  plan_nodes.insert(cur_node);
  while (!_goal_checker(_domain, cur_node->state) &&
         cur_node->best_action.first != nullptr) {
    Value val;
    val.cost(std::get<2>(cur_node->best_action.second->best_parent));
    p.push_back(
        std::make_tuple(cur_node->state, *(cur_node->best_action.first), val));
    cur_node = cur_node->best_action.second;
    if (!plan_nodes.insert(cur_node).second) {
      Logger::error("SKDECIDE exception: cycle detected in the solution plan "
                    "starting in state " +
                    from_state.print());
      throw std::runtime_error("SKDECIDE exception: cycle detected in the "
                               "solution plan starting in state " +
                               from_state.print());
    }
  }
  return p;
}

SK_EHC_SOLVER_TEMPLATE_DECL
typename MapTypeDeducer<typename SK_EHC_SOLVER_CLASS::State,
                        std::pair<typename SK_EHC_SOLVER_CLASS::Action,
                                  typename SK_EHC_SOLVER_CLASS::Value>>::Map
SK_EHC_SOLVER_CLASS::get_policy() const {
  typename MapTypeDeducer<State, std::pair<Action, Value>>::Map p;
  for (auto &n : _graph) {
    if (n.best_action.first != nullptr) {
      Value val;
      val.cost(std::get<2>(n.best_action.second->best_parent));
      p.insert(
          std::make_pair(n.state, std::make_pair(*(n.best_action.first), val)));
    }
  }
  return p;
}

// === Node implementation ===

SK_EHC_SOLVER_TEMPLATE_DECL
SK_EHC_SOLVER_CLASS::Node::Node(const State &s)
    : state(s), best_action({nullptr, nullptr}) {}

SK_EHC_SOLVER_TEMPLATE_DECL
const typename SK_EHC_SOLVER_CLASS::State &
SK_EHC_SOLVER_CLASS::Node::Key::operator()(const Node &n) const {
  return n.state;
}

SK_EHC_SOLVER_TEMPLATE_DECL
template <typename Params>
std::unique_ptr<SK_EHC_SOLVER_CLASS> SK_EHC_SOLVER_CLASS::create_from_params(
    Domain &domain,
    std::function<Predicate(Domain &, const State &)> goal_checker,
    std::function<Value(Domain &, const State &)> heuristic,
    std::function<Value(const State &)> /*terminal_value*/,
    const Params &params, bool verbose) {
  return std::make_unique<EHCSolver>(
      domain, goal_checker, heuristic, PreferredActionsFunctor(nullptr),
      CallbackFunctor([](const EHCSolver &, Domain &) { return false; }),
      params.template get<bool>("verbose", verbose));
}

} // namespace skdecide

#endif // SKDECIDE_EHC_IMPL_HH
