/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_UCB1_ACTION_SELECTOR_IMPL_HH
#define SKDECIDE_MCTS_UCB1_ACTION_SELECTOR_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === UCB1ActionSelector implementation ===

#define SK_MCTS_UCB1_ACTION_SELECTOR_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_UCB1_ACTION_SELECTOR_CLASS UCB1ActionSelector<Tsolver>

SK_MCTS_UCB1_ACTION_SELECTOR_TEMPLATE_DECL
SK_MCTS_UCB1_ACTION_SELECTOR_CLASS::UCB1ActionSelector(double ucb_constant)
    : _ucb_constant(ucb_constant) {}

SK_MCTS_UCB1_ACTION_SELECTOR_TEMPLATE_DECL
SK_MCTS_UCB1_ACTION_SELECTOR_CLASS::UCB1ActionSelector(
    const UCB1ActionSelector &other)
    : _ucb_constant((double)other._ucb_constant) {}

SK_MCTS_UCB1_ACTION_SELECTOR_TEMPLATE_DECL
typename Tsolver::ActionNode *SK_MCTS_UCB1_ACTION_SELECTOR_CLASS::operator()(
    Tsolver &solver, const std::size_t *thread_id,
    const typename Tsolver::StateNode &n) const {
  double best_value = -std::numeric_limits<double>::max();
  typename Tsolver::ActionNode *best_action = nullptr;

  solver.execution_policy().protect(
      [this, &n, &best_value, &best_action, &solver]() {
        for (const auto &a : n.actions) {
          if (a.visits_count > 0) {
            double tentative_value =
                a.value + (2.0 * _ucb_constant *
                           std::sqrt((2.0 * std::log((double)n.visits_count)) /
                                     ((double)a.visits_count)));

            if (tentative_value > best_value) {
              best_value = tentative_value;
              best_action = &const_cast<typename Tsolver::ActionNode &>(
                  a); // we won't change the real key (ActionNode::action) so we
                      // are safe
            }
          }
        }

        if (solver.verbose()) {
          Logger::debug(
              "UCB1 selection from state " + n.state.print() +
              ": value=" + StringConverter::from(best_value) + ", action=" +
              ((best_action != nullptr) ? (best_action->action.print())
                                        : ("nullptr")) +
              Tsolver::ExecutionPolicy::print_thread());
        }
      },
      n.mutex);

  return best_action;
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_UCB1_ACTION_SELECTOR_IMPL_HH
