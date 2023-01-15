/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_DEFAULT_ROLLOUT_POLICY_IMPL_HH
#define SKDECIDE_MCTS_DEFAULT_ROLLOUT_POLICY_IMPL_HH

#include "utils/string_converter.hh"
#include "utils/execution.hh"
#include "utils/logging.hh"

namespace skdecide {

// === DefaultRolloutPolicy implementation ===

#define SK_MCTS_DEFAULT_ROLLOUT_POLICY_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_DEFAULT_ROLLOUT_POLICY_CLASS DefaultRolloutPolicy<Tsolver>

SK_MCTS_DEFAULT_ROLLOUT_POLICY_TEMPLATE_DECL
SK_MCTS_DEFAULT_ROLLOUT_POLICY_CLASS::DefaultRolloutPolicy(
    const PolicyFunctor &policy)
    : _policy(policy) {}

SK_MCTS_DEFAULT_ROLLOUT_POLICY_TEMPLATE_DECL
void SK_MCTS_DEFAULT_ROLLOUT_POLICY_CLASS::operator()(
    Tsolver &solver, const std::size_t *thread_id,
    typename Tsolver::StateNode &n, std::size_t d) const {
  try {
    typename Tsolver::Domain::State current_state;
    bool termination;

    solver.execution_policy().protect(
        [&solver, &n, &current_state, &termination]() {
          if (solver.debug_logs()) {
            Logger::debug("Launching default rollout policy from state " +
                          n.state.print() +
                          Tsolver::ExecutionPolicy::print_thread());
          }
          current_state = n.state;
          termination = n.terminal;
        },
        n.mutex);

    std::size_t current_depth = d;
    double reward = 0.0;
    double gamma_n = 1.0;

    while (!termination && current_depth < solver.max_depth()) {
      typename Tsolver::Domain::Action action =
          _policy(solver.domain(), current_state, thread_id);
      typename Tsolver::Domain::EnvironmentOutcome o =
          solver.transition_mode().random_next_outcome(solver, thread_id,
                                                       current_state, action);
      reward += gamma_n * (o.transition_value().reward());
      gamma_n *= solver.discount();
      current_state = o.observation();
      termination = o.termination();
      current_depth++;
      if (solver.debug_logs()) {
        Logger::debug("Sampled transition: action=" + action.print() +
                      ", next state=" + current_state.print() + ", reward=" +
                      StringConverter::from(o.transition_value().reward()) +
                      Tsolver::ExecutionPolicy::print_thread());
      }
    }

    // since we can come to state n after exhausting the depth, n might be
    // already visited so don't erase its value but rather update it
    solver.execution_policy().protect(
        [&n, &reward]() {
          n.value = ((n.visits_count * n.value) + reward) /
                    ((double)(n.visits_count + 1));
          n.visits_count += 1;
        },
        n.mutex);
  } catch (const std::exception &e) {
    solver.execution_policy().protect(
        [&n, &e]() {
          Logger::error("SKDECIDE exception in MCTS when simulating the random "
                        "default policy from state " +
                        n.state.print() + ": " + e.what() +
                        Tsolver::ExecutionPolicy::print_thread());
        },
        n.mutex);
    throw;
  }
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_DEFAULT_ROLLOUT_POLICY_IMPL_HH
