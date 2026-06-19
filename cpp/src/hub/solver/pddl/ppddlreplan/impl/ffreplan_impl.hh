/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FFREPLAN_IMPL_HH
#define SKDECIDE_FFREPLAN_IMPL_HH

#include "hub/solver/pddl/ppddlreplan/ffreplan.hh"
#include "hub/solver/pddl/ppddlreplan/ff_inner_solver.hh"
#include "hub/solver/pddl/ppddlreplan/impl/ppddlreplan_impl.hh"

namespace skdecide {

namespace pddl {

#define SK_FFREPLAN_TEMPLATE_DECL                                              \
  template <typename Texecution_policy, typename TdeterminizationStrategy>

#define SK_FFREPLAN_CLASS                                                      \
  FFReplanSolver<Texecution_policy, TdeterminizationStrategy>

SK_FFREPLAN_TEMPLATE_DECL
SK_FFREPLAN_CLASS::FFReplanSolver(const Task &task, double dead_end_cost,
                                  std::size_t max_replans,
                                  std::size_t max_steps,
                                  const CallbackFunctor &callback,
                                  bool verbose) {
  typename PPDDL::InnerSolverFactory factory =
      [dead_end_cost, verbose](PddlDeterministicDomain &det_d)
      -> std::unique_ptr<MetaInnerSolverBase<PddlDeterministicDomain>> {
    return std::make_unique<FFInnerSolver<Texecution_policy>>(
        det_d.task(), dead_end_cost, verbose);
  };

  typename PPDDL::CallbackFunctor adapted_callback = [](const PPDDL &) -> bool {
    return false;
  };

  if (callback) {
    _callback_storage = callback;
    adapted_callback = [this](const PPDDL &) -> bool {
      return _callback_storage(*this);
    };
  }

  _impl = std::make_unique<PPDDL>(task, std::move(factory), max_replans,
                                  max_steps, adapted_callback, verbose);
}

SK_FFREPLAN_TEMPLATE_DECL
void SK_FFREPLAN_CLASS::solve(const State &s) { _impl->solve(s); }

SK_FFREPLAN_TEMPLATE_DECL
void SK_FFREPLAN_CLASS::clear() { _impl->clear(); }

SK_FFREPLAN_TEMPLATE_DECL
bool SK_FFREPLAN_CLASS::is_solution_defined_for(const State &s) const {
  return _impl->is_solution_defined_for(s);
}

SK_FFREPLAN_TEMPLATE_DECL
const GroundAction &SK_FFREPLAN_CLASS::get_best_action(const State &s) {
  return _impl->get_best_action(s);
}

SK_FFREPLAN_TEMPLATE_DECL
std::vector<std::pair<State, GroundAction>>
SK_FFREPLAN_CLASS::get_plan() const {
  return _impl->get_plan();
}

SK_FFREPLAN_TEMPLATE_DECL
std::size_t SK_FFREPLAN_CLASS::get_nb_replans() const {
  return _impl->get_nb_replans();
}

SK_FFREPLAN_TEMPLATE_DECL
std::size_t SK_FFREPLAN_CLASS::get_nb_steps() const {
  return _impl->get_nb_steps();
}

SK_FFREPLAN_TEMPLATE_DECL
std::size_t SK_FFREPLAN_CLASS::get_solving_time() const {
  return _impl->get_solving_time();
}

SK_FFREPLAN_TEMPLATE_DECL
double SK_FFREPLAN_CLASS::get_total_cost() const {
  return _impl->get_total_cost();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_FFREPLAN_IMPL_HH
