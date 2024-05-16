/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_DEFAULT_TREE_POLICY_IMPL_HH
#define SKDECIDE_MCTS_DEFAULT_TREE_POLICY_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === DefaultTreePolicy implementation ===

#define SK_MCTS_DEFAULT_TREE_POLICY_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_DEFAULT_TREE_POLICY_CLASS DefaultTreePolicy<Tsolver>

SK_MCTS_DEFAULT_TREE_POLICY_TEMPLATE_DECL
typename Tsolver::StateNode *SK_MCTS_DEFAULT_TREE_POLICY_CLASS::operator()(
    Tsolver &solver,
    const std::size_t *thread_id, // for parallelisation
    const typename Tsolver::Expander &expander,
    const typename Tsolver::ActionSelectorOptimization &action_selector,
    typename Tsolver::StateNode &n, std::size_t &d) const {
  try {
    if (solver.verbose()) {
      solver.execution_policy().protect(
          [&n]() {
            Logger::debug("Launching default tree policy from state " +
                          n.state.print() +
                          Tsolver::ExecutionPolicy::print_thread());
          },
          n.mutex);
    }

    solver.transition_mode().init_rollout(solver, thread_id);
    typename Tsolver::StateNode *current_node = &n;

    while (!(current_node->terminal) && d < solver.max_depth()) {
      typename Tsolver::StateNode *next_node =
          expander(solver, thread_id, *current_node);
      d++;

      if (next_node == nullptr) { // node fully expanded
        typename Tsolver::ActionNode *action =
            action_selector(solver, thread_id, *current_node);

        if (action == nullptr) {
          // It might happen in parallel execution mode when the current node's
          // actions are all being expanded by concurrent threads that claim the
          // node is expanded but not yet backpropagated and a new thread
          // meantime comes and sees the node as expanded, thus all action
          // visits counts are still equal to zero (implying action_selector to
          // return nullptr). This shall NOT happen in sequential execution
          // mode.
          break;
        } else {
          next_node = solver.transition_mode().random_next_node(
              solver, thread_id, *action);

          if (next_node == nullptr) { // might happen with step transition mode
                                      // and stochastic environments
            break;
          } else {
            current_node = next_node;
          }
        }
      } else {
        current_node = next_node;
        break;
      }
    }

    return current_node;
  } catch (const std::exception &e) {
    solver.execution_policy().protect(
        [&n, &e]() {
          Logger::error("SKDECIDE exception in MCTS when simulating the tree "
                        "policy from state " +
                        n.state.print() + ": " + e.what() +
                        Tsolver::ExecutionPolicy::print_thread());
        },
        n.mutex);
    throw;
  }
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_DEFAULT_TREE_POLICY_IMPL_HH
