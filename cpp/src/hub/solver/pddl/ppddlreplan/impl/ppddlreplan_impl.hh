/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PPDDLREPLAN_IMPL_HH
#define SKDECIDE_PPDDLREPLAN_IMPL_HH

#include "hub/solver/pddl/ppddlreplan/ppddlreplan.hh"
#include "hub/solver/sspreplan/impl/sspreplan_impl.hh"

namespace skdecide {

namespace pddl {

#define SK_PPDDLREPLAN_TEMPLATE_DECL                                           \
  template <typename Texecution_policy, typename TdeterminizationStrategy>

#define SK_PPDDLREPLAN_CLASS                                                   \
  PPDDLReplanSolver<Texecution_policy, TdeterminizationStrategy>

SK_PPDDLREPLAN_TEMPLATE_DECL
SK_PPDDLREPLAN_CLASS::PPDDLReplanSolver(const Task &task,
                                        InnerSolverFactory inner_solver_factory,
                                        std::size_t max_replans,
                                        std::size_t max_steps,
                                        const CallbackFunctor &callback,
                                        bool verbose) {
  _stochastic_domain = std::make_unique<Domain>(task);

  Adapter adapter(task);

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
      *_stochastic_domain, std::move(adapter), std::move(inner_solver_factory),
      goal_checker, max_replans, max_steps, callback_adapted, verbose);
}

SK_PPDDLREPLAN_TEMPLATE_DECL
void SK_PPDDLREPLAN_CLASS::solve(const State &s) {
  _solver->solve(PddlState(s));
}

SK_PPDDLREPLAN_TEMPLATE_DECL
void SK_PPDDLREPLAN_CLASS::clear() { _solver->clear(); }

SK_PPDDLREPLAN_TEMPLATE_DECL
bool SK_PPDDLREPLAN_CLASS::is_solution_defined_for(const State &s) const {
  return _solver->is_solution_defined_for(PddlState(s));
}

SK_PPDDLREPLAN_TEMPLATE_DECL
const GroundAction &SK_PPDDLREPLAN_CLASS::get_best_action(const State &s) {
  return _solver->get_best_action(PddlState(s));
}

SK_PPDDLREPLAN_TEMPLATE_DECL
std::vector<std::pair<State, GroundAction>>
SK_PPDDLREPLAN_CLASS::get_plan() const {
  const auto &ssp_plan = _solver->get_plan();
  std::vector<std::pair<State, GroundAction>> result;
  result.reserve(ssp_plan.size());
  for (const auto &[s, a] : ssp_plan) {
    result.emplace_back(static_cast<const State &>(s),
                        static_cast<const GroundAction &>(a));
  }
  return result;
}

SK_PPDDLREPLAN_TEMPLATE_DECL
std::size_t SK_PPDDLREPLAN_CLASS::get_nb_replans() const {
  return _solver->get_nb_replans();
}

SK_PPDDLREPLAN_TEMPLATE_DECL
std::size_t SK_PPDDLREPLAN_CLASS::get_nb_steps() const {
  return _solver->get_nb_steps();
}

SK_PPDDLREPLAN_TEMPLATE_DECL
std::size_t SK_PPDDLREPLAN_CLASS::get_solving_time() const {
  return _solver->get_solving_time();
}

SK_PPDDLREPLAN_TEMPLATE_DECL
double SK_PPDDLREPLAN_CLASS::get_total_cost() const {
  return _solver->get_total_cost();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PPDDLREPLAN_IMPL_HH
