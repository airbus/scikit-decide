/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PY_MCTS_INST_HH
#define SKDECIDE_PY_MCTS_INST_HH

namespace skdecide {

template <typename... TypeInstantiations>
template <template <typename...> class... TemplateInstantiations>
PyMCTSSolver::PartialSolverInstantiator::TypeList<TypeInstantiations...>::
    TemplateList<TemplateInstantiations...>::Instantiate::Instantiate(
        PartialSolverInstantiator &This, MCTS_SOLVER_DECL_ARGS) {
  TemplateInstantiator::select(
      ActionSelector(This._action_selector_optimization),
      ActionSelector(This._action_selector_execution),
      RolloutPolicySelector(This._rollout_policy, This._custom_policy_functor,
                            This._heuristic_functor),
      BackPropagatorSelector(This._back_propagator),
      typename FullSolverInstantiator::
          template TypeList<TypeInstantiations...>::template TemplateList<
              TemplateInstantiations...>(This._implementation))
      .instantiate(MCTS_SOLVER_ARGS);
}

template <typename... PartialTypeInstantiations>
template <template <typename...> class... PartialTemplateInstantiations>
template <template <typename...> class... TemplateInstantiations>
PyMCTSSolver::FullSolverInstantiator::TypeList<PartialTypeInstantiations...>::
    TemplateList<PartialTemplateInstantiations...>::Instantiate<
        TemplateInstantiations...>::Instantiate(TemplateList &This,
                                                MCTS_SOLVER_DECL_ARGS) {
  This._implementation =
      std::make_unique<Implementation<PartialTypeInstantiations...,
                                      PartialTemplateInstantiations...,
                                      TemplateInstantiations...>>(
          MCTS_SOLVER_ARGS);
}

} // namespace skdecide

#endif // SKDECIDE_PY_MCTS_INST_HH
