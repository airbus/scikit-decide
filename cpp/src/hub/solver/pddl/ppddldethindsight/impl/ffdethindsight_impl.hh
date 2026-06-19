/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_FFDETHINDSIGHT_IMPL_HH
#define SKDECIDE_FFDETHINDSIGHT_IMPL_HH

#include "hub/solver/pddl/ppddldethindsight/ffdethindsight.hh"
#include "hub/solver/pddl/ppddlreplan/ff_inner_solver.hh"
#include "hub/solver/pddl/ppddldethindsight/impl/ppddldethindsight_impl.hh"

namespace skdecide {

namespace pddl {

#define SK_FFDETHINDSIGHT_TEMPLATE_DECL template <typename Texecution_policy>

#define SK_FFDETHINDSIGHT_CLASS FFDetHindsightSolver<Texecution_policy>

SK_FFDETHINDSIGHT_TEMPLATE_DECL
SK_FFDETHINDSIGHT_CLASS::FFDetHindsightSolver(
    const Task &task, std::size_t sample_width, double dead_end_cost,
    std::size_t max_steps, double discount, double epsilon,
    const CallbackFunctor &callback, bool verbose) {
  typename PPDDLDetHindsightSolver<Texecution_policy>::InnerSolverFactory
      factory = [dead_end_cost, verbose](PddlDeterministicDomain &det_d)
      -> std::unique_ptr<MetaInnerSolverBase<PddlDeterministicDomain>> {
    return std::make_unique<FFInnerSolver<SequentialExecution>>(
        det_d.task(), dead_end_cost, verbose);
  };

  typename PPDDLDetHindsightSolver<Texecution_policy>::CallbackFunctor
      adapted_callback =
          [](const PPDDLDetHindsightSolver<Texecution_policy> &) -> bool {
    return false;
  };

  if (callback) {
    _callback_storage = callback;
    adapted_callback =
        [this](const PPDDLDetHindsightSolver<Texecution_policy> &) -> bool {
      return _callback_storage(*this);
    };
  }

  _impl = std::make_unique<PPDDLDetHindsightSolver<Texecution_policy>>(
      task, std::move(factory), sample_width, dead_end_cost, max_steps,
      discount, epsilon, adapted_callback, verbose);
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
void SK_FFDETHINDSIGHT_CLASS::solve(const State &s) { _impl->solve(s); }

SK_FFDETHINDSIGHT_TEMPLATE_DECL
void SK_FFDETHINDSIGHT_CLASS::clear() { _impl->clear(); }

SK_FFDETHINDSIGHT_TEMPLATE_DECL
bool SK_FFDETHINDSIGHT_CLASS::is_solution_defined_for(const State &s) const {
  return _impl->is_solution_defined_for(s);
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
const GroundAction &SK_FFDETHINDSIGHT_CLASS::get_best_action(const State &s) {
  return _impl->get_best_action(s);
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
double SK_FFDETHINDSIGHT_CLASS::get_best_value(const State &s) const {
  return _impl->get_best_value(s);
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
std::size_t SK_FFDETHINDSIGHT_CLASS::get_nb_steps() const {
  return _impl->get_nb_steps();
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
std::size_t SK_FFDETHINDSIGHT_CLASS::get_solving_time() const {
  return _impl->get_solving_time();
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_FFDETHINDSIGHT_CLASS::get_explored_states() const {
  return _impl->get_explored_states();
}

SK_FFDETHINDSIGHT_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_FFDETHINDSIGHT_CLASS::get_terminal_states() const {
  return _impl->get_terminal_states();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_FFDETHINDSIGHT_IMPL_HH
