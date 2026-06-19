/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_RFF_IMPL_HH
#define SKDECIDE_RFF_IMPL_HH

#include "hub/solver/pddl/ppddlplanmerger/rff.hh"
#include "hub/solver/pddl/ppddlreplan/ff_inner_solver.hh"
#include "hub/solver/pddl/ppddlplanmerger/impl/ppddlplanmerger_impl.hh"

namespace skdecide {

namespace pddl {

#define SK_RFF_TEMPLATE_DECL                                                   \
  template <typename Texecution_policy, typename TdeterminizationStrategy>

#define SK_RFF_CLASS RFFSolver<Texecution_policy, TdeterminizationStrategy>

SK_RFF_TEMPLATE_DECL
SK_RFF_CLASS::RFFSolver(const Task &task, double dead_end_cost, double rho,
                        std::size_t mc_samples, std::size_t max_iterations,
                        std::size_t max_steps, bool optimize_policy_graph,
                        double discount, double epsilon,
                        const CallbackFunctor &callback, bool verbose) {
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

  _impl = std::make_unique<PPDDL>(task, std::move(factory), rho, mc_samples,
                                  max_iterations, max_steps, dead_end_cost,
                                  optimize_policy_graph, discount, epsilon,
                                  adapted_callback, verbose);
}

SK_RFF_TEMPLATE_DECL
void SK_RFF_CLASS::solve(const State &s) { _impl->solve(s); }

SK_RFF_TEMPLATE_DECL
void SK_RFF_CLASS::resolve(const State &s) { _impl->resolve(s); }

SK_RFF_TEMPLATE_DECL
void SK_RFF_CLASS::clear() { _impl->clear(); }

SK_RFF_TEMPLATE_DECL
bool SK_RFF_CLASS::is_solution_defined_for(const State &s) const {
  return _impl->is_solution_defined_for(s);
}

SK_RFF_TEMPLATE_DECL
const GroundAction &SK_RFF_CLASS::get_best_action(const State &s) const {
  return _impl->get_best_action(s);
}

SK_RFF_TEMPLATE_DECL
double SK_RFF_CLASS::get_best_value(const State &s) const {
  return _impl->get_best_value(s);
}

SK_RFF_TEMPLATE_DECL
std::size_t SK_RFF_CLASS::get_nb_iterations() const {
  return _impl->get_nb_iterations();
}

SK_RFF_TEMPLATE_DECL
std::size_t SK_RFF_CLASS::get_nb_plans() const { return _impl->get_nb_plans(); }

SK_RFF_TEMPLATE_DECL
std::size_t SK_RFF_CLASS::get_solving_time() const {
  return _impl->get_solving_time();
}

SK_RFF_TEMPLATE_DECL
std::size_t SK_RFF_CLASS::get_policy_size() const {
  return _impl->get_policy_size();
}

SK_RFF_TEMPLATE_DECL
typename SK_RFF_CLASS::PPDDL::Solver::PolicyMap
SK_RFF_CLASS::get_policy() const {
  return _impl->get_policy();
}

SK_RFF_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_RFF_CLASS::get_explored_states() const {
  return _impl->get_explored_states();
}

SK_RFF_TEMPLATE_DECL
typename SetTypeDeducer<PddlState>::Set
SK_RFF_CLASS::get_terminal_states() const {
  return _impl->get_terminal_states();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_RFF_IMPL_HH
