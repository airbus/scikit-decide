/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FF_IMPL_HH
#define SKDECIDE_PDDL_FF_IMPL_HH

#include <stdexcept>

#include "hub/solver/ehc/impl/ehc_impl.hh"
#include "utils/logging.hh"

namespace skdecide {

namespace pddl {

#define SK_PDDL_FF_TEMPLATE_DECL template <typename Texecution_policy>

#define SK_PDDL_FF_CLASS FFSolver<Texecution_policy>

SK_PDDL_FF_TEMPLATE_DECL
SK_PDDL_FF_CLASS::FFSolver(const Task &task, double dead_end_cost,
                           const CallbackFunctor &callback, bool verbose)
    : _task(task), _dead_end_cost(dead_end_cost), _callback(callback),
      _verbose(verbose) {

  _heuristic = std::make_unique<FFHeuristic>(task, 1.0, dead_end_cost, verbose);
  _domain = std::make_unique<PddlDeterministicDomain>(task);
  _goal_checker = std::make_unique<GoalChecker>(task);

  auto goal_functor = [this](PddlDeterministicDomain &,
                             const PddlState &s) -> bool {
    return _goal_checker->is_goal(s);
  };

  auto heuristic_functor = [this](PddlDeterministicDomain &,
                                  const PddlState &s) -> PddlValue {
    ensure_computed(s);
    return PddlValue(_last_result.first);
  };

  auto preferred_functor =
      [this](PddlDeterministicDomain &,
             const PddlState &s) -> std::vector<PddlAction> {
    ensure_computed(s);
    std::vector<PddlAction> result;
    result.reserve(_last_result.second.size());
    for (auto &ga : _last_result.second) {
      result.emplace_back(ga);
    }
    return result;
  };

  auto callback_functor =
      [this](const EHCSolver<PddlDeterministicDomain, Texecution_policy> &,
             PddlDeterministicDomain &) -> bool { return _callback(*this); };

  _ehc =
      std::make_unique<EHCSolver<PddlDeterministicDomain, Texecution_policy>>(
          *_domain, goal_functor, heuristic_functor, preferred_functor,
          callback_functor, verbose);
}

SK_PDDL_FF_TEMPLATE_DECL
void SK_PDDL_FF_CLASS::ensure_computed(const PddlState &s) const {
  if (!_has_cached || !(PddlState::Equal()(_last_state, s))) {
    _last_result = _heuristic->compute_with_helpful(s);
    _last_state = s;
    _has_cached = true;
  }
}

SK_PDDL_FF_TEMPLATE_DECL
void SK_PDDL_FF_CLASS::solve(const State &initial_state) {
  _initial_state = PddlState(initial_state);
  _has_cached = false;
  _ehc->solve(_initial_state);
}

SK_PDDL_FF_TEMPLATE_DECL
void SK_PDDL_FF_CLASS::clear() {
  _ehc->clear();
  _has_cached = false;
}

SK_PDDL_FF_TEMPLATE_DECL
bool SK_PDDL_FF_CLASS::is_solution_defined_for(const State &s) const {
  return _ehc->is_solution_defined_for(PddlState(s));
}

SK_PDDL_FF_TEMPLATE_DECL
const GroundAction &SK_PDDL_FF_CLASS::get_best_action(const State &s) const {
  return _ehc->get_best_action(PddlState(s));
}

SK_PDDL_FF_TEMPLATE_DECL
std::vector<std::pair<State, GroundAction>> SK_PDDL_FF_CLASS::get_plan() const {
  auto ehc_plan = _ehc->get_plan(_initial_state);
  std::vector<std::pair<State, GroundAction>> result;
  result.reserve(ehc_plan.size());
  for (auto &[state, action, value] : ehc_plan) {
    result.emplace_back(static_cast<const State &>(state),
                        static_cast<const GroundAction &>(action));
  }
  return result;
}

SK_PDDL_FF_TEMPLATE_DECL
std::size_t SK_PDDL_FF_CLASS::get_nb_explored_states() const {
  return _ehc->get_nb_explored_states();
}

SK_PDDL_FF_TEMPLATE_DECL
std::vector<State> SK_PDDL_FF_CLASS::get_explored_states() const {
  auto ehc_states = _ehc->get_explored_states();
  std::vector<State> result;
  result.reserve(ehc_states.size());
  for (auto &s : ehc_states) {
    result.push_back(static_cast<const State &>(s));
  }
  return result;
}

SK_PDDL_FF_TEMPLATE_DECL
std::size_t SK_PDDL_FF_CLASS::get_solving_time() const {
  return _ehc->get_solving_time();
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_FF_IMPL_HH
