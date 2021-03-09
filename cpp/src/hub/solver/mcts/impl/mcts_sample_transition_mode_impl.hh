/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_MCTS_SAMPLE_TRANSITION_MODE_IMPL_HH
#define SKDECIDE_MCTS_SAMPLE_TRANSITION_MODE_IMPL_HH

namespace skdecide {

// === SampleTransitionMode implementation ===

#define SK_MCTS_SAMPLE_TRANSITION_MODE_TEMPLATE_DECL \
template <typename Tsolver>

#define SK_MCTS_SAMPLE_TRANSITION_MODE_CLASS \
SampleTransitionMode<Tsolver>

SK_MCTS_SAMPLE_TRANSITION_MODE_TEMPLATE_DECL
void SK_MCTS_SAMPLE_TRANSITION_MODE_CLASS::init_rollout(Tsolver& solver,
                                                        const std::size_t* thread_id) const {}

SK_MCTS_SAMPLE_TRANSITION_MODE_TEMPLATE_DECL
typename Tsolver::Domain::EnvironmentOutcome
SK_MCTS_SAMPLE_TRANSITION_MODE_CLASS::random_next_outcome(Tsolver& solver,
                                                          const std::size_t* thread_id,
                                                          const typename Tsolver::Domain::State& state,
                                                          const typename Tsolver::Domain::Action& action) const {
    return solver.domain().sample(state, action, thread_id);
}

SK_MCTS_SAMPLE_TRANSITION_MODE_TEMPLATE_DECL
typename Tsolver::StateNode*
SK_MCTS_SAMPLE_TRANSITION_MODE_CLASS::random_next_node(Tsolver& solver,
                                                       const std::size_t* thread_id,
                                                       typename Tsolver::ActionNode& action) const {
    typename Tsolver::StateNode* n = nullptr;

    solver.execution_policy().protect([&n, &action, &solver](){
        solver.execution_policy().protect([&n, &action, &solver](){
            n = action.dist_to_outcome[action.dist(solver.gen())]->first;
        }, solver.gen_mutex());
    }, action.parent->mutex);

    return n;
}

} // namespace skdecide

#endif // SKDECIDE_MCTS_SAMPLE_TRANSITION_MODE_IMPL_HH
