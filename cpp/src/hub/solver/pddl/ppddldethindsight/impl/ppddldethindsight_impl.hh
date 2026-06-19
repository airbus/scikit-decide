/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PPDDLDETHINDSIGHT_IMPL_HH
#define SKDECIDE_PPDDLDETHINDSIGHT_IMPL_HH

#include "hub/solver/pddl/ppddldethindsight/ppddldethindsight.hh"
#include "hub/solver/sspdethindsight/impl/sspdethindsight_impl.hh"
#include "hub/solver/pddl/determinization/impl/pddl_determinization_adapter_impl.hh"

namespace skdecide {

namespace pddl {

#define SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL template <typename Texecution_policy>

#define SK_PPDDLDETHINDSIGHT_CLASS PPDDLDetHindsightSolver<Texecution_policy>

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
SK_PPDDLDETHINDSIGHT_CLASS::PPDDLDetHindsightSolver(
    const Task &task, InnerSolverFactory inner_solver_factory,
    std::size_t sample_width, double dead_end_cost, std::size_t max_steps,
    double discount, double epsilon, const CallbackFunctor &callback,
    bool verbose)
    : _task(task) {
  _stochastic_domain = std::make_unique<Domain>(task);

  auto adapter_factory = [&task]() -> Adapter { return Adapter(task); };

  GoalChecker gc(task);
  auto goal_checker = [gc = std::move(gc)](PddlDomainForDeterminization &,
                                           const PddlState &s) -> bool {
    return gc.is_goal(s);
  };

  auto callback_adapted =
      [this, callback](const Solver &, PddlDomainForDeterminization &) -> bool {
    return callback(*this);
  };

  _solver = std::make_unique<Solver>(
      *_stochastic_domain, std::move(inner_solver_factory),
      std::move(adapter_factory), goal_checker, sample_width, dead_end_cost,
      max_steps, discount, epsilon, callback_adapted, verbose);
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
void SK_PPDDLDETHINDSIGHT_CLASS::solve(const State &s) {
  _solver->solve(PddlState(s));
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
void SK_PPDDLDETHINDSIGHT_CLASS::clear() { _solver->clear(); }

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
bool SK_PPDDLDETHINDSIGHT_CLASS::is_solution_defined_for(const State &s) const {
  return _solver->is_solution_defined_for(PddlState(s));
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
const GroundAction &
SK_PPDDLDETHINDSIGHT_CLASS::get_best_action(const State &s) {
  return _solver->get_best_action(PddlState(s));
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
double SK_PPDDLDETHINDSIGHT_CLASS::get_best_value(const State &s) const {
  return _solver->get_best_value(PddlState(s));
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
std::size_t SK_PPDDLDETHINDSIGHT_CLASS::get_nb_steps() const {
  return _solver->get_nb_steps();
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
std::size_t SK_PPDDLDETHINDSIGHT_CLASS::get_solving_time() const {
  return _solver->get_solving_time();
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_PPDDLDETHINDSIGHT_CLASS::get_explored_states() const {
  return _solver->get_explored_states();
}

SK_PPDDLDETHINDSIGHT_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_PPDDLDETHINDSIGHT_CLASS::get_terminal_states() const {
  return _solver->get_terminal_states();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PPDDLDETHINDSIGHT_IMPL_HH
