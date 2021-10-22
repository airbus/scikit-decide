/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_PARTIAL_EXPAND_IMPL_HH
#define SKDECIDE_MCTS_PARTIAL_EXPAND_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === PartialExpand implementation ===

#define SK_MCTS_PARTIAL_EXPAND_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_PARTIAL_EXPAND_CLASS PartialExpand<Tsolver>

SK_MCTS_PARTIAL_EXPAND_TEMPLATE_DECL
SK_MCTS_PARTIAL_EXPAND_CLASS::PartialExpand(const double &state_expansion_rate,
                                            const double &action_expansion_rate,
                                            const HeuristicFunctor &heuristic)
    : _heuristic(heuristic), _state_expansion_rate(state_expansion_rate),
      _action_expansion_rate(action_expansion_rate) {}

SK_MCTS_PARTIAL_EXPAND_TEMPLATE_DECL
SK_MCTS_PARTIAL_EXPAND_CLASS::PartialExpand(const PartialExpand &other)
    : _heuristic(other._heuristic),
      _state_expansion_rate(other._state_expansion_rate),
      _action_expansion_rate(other._action_expansion_rate) {}

SK_MCTS_PARTIAL_EXPAND_TEMPLATE_DECL
typename Tsolver::StateNode *
SK_MCTS_PARTIAL_EXPAND_CLASS::operator()(Tsolver &solver,
                                         const std::size_t *thread_id,
                                         typename Tsolver::StateNode &n) const {
  try {
    if (solver.debug_logs()) {
      solver.execution_policy().protect(
          [&n]() {
            Logger::debug("Testing expansion of state " + n.state.print() +
                          Tsolver::ExecutionPolicy::print_thread());
          },
          n.mutex);
    }

    // Sample an action
    std::bernoulli_distribution dist_state_expansion(
        std::exp(-_state_expansion_rate * n.expansions_count));
    typename Tsolver::ActionNode *action_node = nullptr;
    bool dist_res = false;

    solver.execution_policy().protect(
        [&dist_res, &solver, &dist_state_expansion]() {
          dist_res = dist_state_expansion(solver.gen());
        },
        solver.gen_mutex());

    if (dist_res) {
      typename Tsolver::Domain::Action action =
          solver.domain().get_applicable_actions(n.state, thread_id).sample();
      solver.execution_policy().protect(
          [&n, &action, &action_node, &solver]() {
            auto a = n.actions.emplace(typename Tsolver::ActionNode(action));
            action_node = &const_cast<typename Tsolver::ActionNode &>(
                *(a.first)); // we won't change the real key
                             // (ActionNode::action) so we are safe

            if (a.second) { // new action
              n.expansions_count += 1;
              action_node->parent = &n;
            }

            if (solver.debug_logs()) {
              Logger::debug(
                  "Sampled a new action: " + action_node->action.print() +
                  Tsolver::ExecutionPolicy::print_thread());
            }
          },
          n.mutex);
    } else {
      std::vector<typename Tsolver::ActionNode *> actions;

      solver.execution_policy().protect(
          [&n, &actions]() {
            for (auto &a : n.actions) {
              actions.push_back(&const_cast<typename Tsolver::ActionNode &>(
                  a)); // we won't change the real key (ActionNode::action) so
                       // we are safe
            }
          },
          n.mutex);

      std::uniform_int_distribution<std::size_t> dist_known_actions(
          0, actions.size() - 1);
      std::size_t action_id = 0;

      solver.execution_policy().protect(
          [&action_id, &solver, &dist_known_actions]() {
            action_id = dist_known_actions(solver.gen());
          },
          solver.gen_mutex());

      action_node = actions[action_id];
      if (solver.debug_logs()) {
        solver.execution_policy().protect(
            [&action_node]() {
              Logger::debug("Sampled among known actions: " +
                            action_node->action.print() +
                            Tsolver::ExecutionPolicy::print_thread());
            },
            action_node->parent->mutex);
      }
    }

    // Sample an outcome
    std::bernoulli_distribution dist_action_expansion(
        std::exp(-_action_expansion_rate * (action_node->expansions_count)));
    typename Tsolver::StateNode *ns = nullptr;

    solver.execution_policy().protect(
        [&dist_res, &solver, &dist_action_expansion]() {
          dist_res = dist_action_expansion(solver.gen());
        },
        solver.gen_mutex());

    if (dist_res) {
      typename Tsolver::Domain::EnvironmentOutcome to =
          solver.transition_mode().random_next_outcome(
              solver, thread_id, n.state, action_node->action);
      std::pair<typename Tsolver::Graph::iterator, bool> s;

      solver.execution_policy().protect([&s, &solver, &to]() {
        s = solver.graph().emplace(to.observation());
      });

      ns = &const_cast<typename Tsolver::StateNode &>(
          *(s.first)); // we won't change the real key (StateNode::state) so we
                       // are safe

      if (s.second) { // new state
        solver.execution_policy().protect(
            [this, &ns, &to, &solver, &thread_id]() {
              ns->terminal = to.termination();
              std::pair<typename Tsolver::Domain::Value, std::size_t> h =
                  _heuristic(solver.domain(), ns->state, thread_id);
              ns->value = h.first.reward();
              ns->visits_count = h.second;
            },
            ns->mutex);
      }

      std::pair<typename Tsolver::ActionNode::OutcomeMap::iterator, bool> ins;
      solver.execution_policy().protect(
          [&action_node, &ns, &to, &ins]() {
            ins = action_node->outcomes.emplace(std::make_pair(
                ns, std::make_pair(to.transition_value().reward(), 1)));
          },
          action_node->parent->mutex);

      // Update the outcome's reward and visits count
      if (ins.second) { // new outcome
        solver.execution_policy().protect(
            [&action_node, &ins]() {
              action_node->dist_to_outcome.push_back(ins.first);
              action_node->expansions_count += 1;
            },
            action_node->parent->mutex);

        solver.execution_policy().protect(
            [&ns, &action_node]() { ns->parents.insert(action_node); },
            ns->mutex);
      } else { // known outcome
        solver.execution_policy().protect(
            [&ins, &to, &ns]() {
              std::pair<double, std::size_t> &mp = ins.first->second;
              mp.first = ((double)(mp.second * mp.first) +
                          to.transition_value().reward()) /
                         ((double)(mp.second + 1));
              mp.second += 1;
              ns = nullptr; // we have not discovered anything new
            },
            action_node->parent->mutex);
      }

      // Reconstruct the probability distribution
      solver.execution_policy().protect(
          [&action_node]() {
            std::vector<double> weights(action_node->dist_to_outcome.size());

            for (unsigned int oid = 0; oid < weights.size(); oid++) {
              weights[oid] =
                  (double)action_node->dist_to_outcome[oid]->second.second;
            }

            action_node->dist =
                std::discrete_distribution<>(weights.begin(), weights.end());
          },
          action_node->parent->mutex);
    } else {
      ns = nullptr; // we have not discovered anything new
    }

    if (solver.debug_logs()) {
      if (ns) {
        solver.execution_policy().protect(
            [&ns]() {
              Logger::debug("Sampled a new outcome: " + ns->state.print() +
                            Tsolver::ExecutionPolicy::print_thread());
            },
            ns->mutex);
      } else {
        solver.execution_policy().protect(
            [&n]() {
              Logger::debug("Not expanding state: " + n.state.print() +
                            Tsolver::ExecutionPolicy::print_thread());
            },
            n.mutex);
      }
    }

    return ns;
  } catch (const std::exception &e) {
    solver.execution_policy().protect(
        [&n, &e]() {
          Logger::error("SKDECIDE exception in MCTS when expanding state " +
                        n.state.print() + ": " + e.what() +
                        Tsolver::ExecutionPolicy::print_thread());
        },
        n.mutex);
    throw;
  }
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_PARTIAL_EXPAND_IMPL_HH
