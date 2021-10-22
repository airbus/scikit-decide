/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_STEP_TRANSITION_MODE_IMPL_HH
#define SKDECIDE_MCTS_STEP_TRANSITION_MODE_IMPL_HH

namespace skdecide {

// === StepTransitionMode implementation ===

#define SK_MCTS_STEP_TRANSITION_MODE_TEMPLATE_DECL template <typename Tsolver>

#define SK_MCTS_STEP_TRANSITION_MODE_CLASS StepTransitionMode<Tsolver>

SK_MCTS_STEP_TRANSITION_MODE_TEMPLATE_DECL
void SK_MCTS_STEP_TRANSITION_MODE_CLASS::init_rollout(
    Tsolver &solver, const std::size_t *thread_id) const {
  solver.domain().reset(thread_id);
  std::for_each(
      solver.action_prefix().begin(), solver.action_prefix().end(),
      [&solver, &thread_id](const typename Tsolver::Domain::Action &a) {
        solver.domain().step(a, thread_id);
      });
}

SK_MCTS_STEP_TRANSITION_MODE_TEMPLATE_DECL
typename Tsolver::Domain::EnvironmentOutcome
SK_MCTS_STEP_TRANSITION_MODE_CLASS::random_next_outcome(
    Tsolver &solver, const std::size_t *thread_id,
    const typename Tsolver::Domain::State &state,
    const typename Tsolver::Domain::Action &action) const {
  return solver.domain().step(action, thread_id);
}

SK_MCTS_STEP_TRANSITION_MODE_TEMPLATE_DECL
typename Tsolver::StateNode *
SK_MCTS_STEP_TRANSITION_MODE_CLASS::random_next_node(
    Tsolver &solver, const std::size_t *thread_id,
    typename Tsolver::ActionNode &action) const {
  auto outcome = solver.domain().step(action.action, thread_id);
  typename Tsolver::StateNode *n = nullptr;

  solver.execution_policy().protect([&n, &solver, &outcome]() {
    auto si =
        solver.graph().find(typename Tsolver::StateNode(outcome.observation()));
    if (si != solver.graph().end()) {
      // we won't change the real key (ActionNode::action) so we are safe
      n = &const_cast<typename Tsolver::StateNode &>(*si);
    }
  });

  return n;
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_STEP_TRANSITION_MODE_IMPL_HH
