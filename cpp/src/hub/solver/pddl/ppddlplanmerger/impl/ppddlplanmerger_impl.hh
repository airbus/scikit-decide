/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PPDDLPLANMERGER_IMPL_HH
#define SKDECIDE_PPDDLPLANMERGER_IMPL_HH

#include "hub/solver/pddl/ppddlplanmerger/ppddlplanmerger.hh"
#include "hub/solver/sspplanmerger/impl/sspplanmerger_impl.hh"
#include "hub/solver/pddl/determinization/impl/pddl_determinization_adapter_impl.hh"

namespace skdecide {

namespace pddl {

#define SK_PPDDLPLANMERGER_TEMPLATE_DECL                                       \
  template <typename Texecution_policy, typename TdeterminizationStrategy>

#define SK_PPDDLPLANMERGER_CLASS                                               \
  PPDDLPlanMergerSolver<Texecution_policy, TdeterminizationStrategy>

SK_PPDDLPLANMERGER_TEMPLATE_DECL
SK_PPDDLPLANMERGER_CLASS::PPDDLPlanMergerSolver(
    const Task &task, InnerSolverFactory inner_solver_factory, double rho,
    std::size_t mc_samples, std::size_t max_iterations, std::size_t max_steps,
    double dead_end_cost, bool optimize_policy_graph, double discount,
    double epsilon, const CallbackFunctor &callback, bool verbose)
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
      *_stochastic_domain, std::move(adapter_factory),
      std::move(inner_solver_factory), goal_checker, rho, mc_samples,
      max_iterations, max_steps, dead_end_cost, optimize_policy_graph, discount,
      epsilon, callback_adapted, verbose);
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
void SK_PPDDLPLANMERGER_CLASS::solve(const State &s) {
  _solver->solve(PddlState(s));
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
void SK_PPDDLPLANMERGER_CLASS::resolve(const State &s) {
  _solver->resolve(PddlState(s));
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
void SK_PPDDLPLANMERGER_CLASS::clear() { _solver->clear(); }

SK_PPDDLPLANMERGER_TEMPLATE_DECL
bool SK_PPDDLPLANMERGER_CLASS::is_solution_defined_for(const State &s) const {
  return _solver->is_solution_defined_for(PddlState(s));
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
const GroundAction &
SK_PPDDLPLANMERGER_CLASS::get_best_action(const State &s) const {
  return _solver->get_best_action(PddlState(s));
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
double SK_PPDDLPLANMERGER_CLASS::get_best_value(const State &s) const {
  return _solver->get_best_value(PddlState(s));
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
std::size_t SK_PPDDLPLANMERGER_CLASS::get_nb_iterations() const {
  return _solver->get_nb_iterations();
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
std::size_t SK_PPDDLPLANMERGER_CLASS::get_nb_plans() const {
  return _solver->get_nb_plans();
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
std::size_t SK_PPDDLPLANMERGER_CLASS::get_solving_time() const {
  return _solver->get_solving_time();
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
std::size_t SK_PPDDLPLANMERGER_CLASS::get_policy_size() const {
  return _solver->get_policy_size();
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
typename SK_PPDDLPLANMERGER_CLASS::Solver::PolicyMap
SK_PPDDLPLANMERGER_CLASS::get_policy() const {
  return _solver->get_policy();
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_PPDDLPLANMERGER_CLASS::get_explored_states() const {
  return _solver->get_explored_states();
}

SK_PPDDLPLANMERGER_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_PPDDLPLANMERGER_CLASS::get_terminal_states() const {
  return _solver->get_terminal_states();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PPDDLPLANMERGER_IMPL_HH
