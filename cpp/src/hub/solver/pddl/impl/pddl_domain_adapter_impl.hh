/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DOMAIN_ADAPTER_IMPL_HH
#define SKDECIDE_PDDL_DOMAIN_ADAPTER_IMPL_HH

namespace skdecide {

namespace pddl {

inline PddlDeterministicDomain::PddlDeterministicDomain(const Task &task)
    : _task(task), _aops_gen(task), _succ_gen(task) {
  _total_cost_idx = task.total_cost_function();
}

inline PddlDeterministicDomain::ActionSpace
PddlDeterministicDomain::get_applicable_actions(const State &s) const {
  auto ga_list = _aops_gen.get_applicable_actions(s, false);
  ActionSpace result;
  result._actions.reserve(ga_list.size());
  for (auto &ga : ga_list) {
    result._actions.emplace_back(std::move(ga));
  }
  return result;
}

inline PddlDeterministicDomain::State
PddlDeterministicDomain::get_next_state(const State &s, const Action &a) const {
  auto succs = _succ_gen.get_successors(s, a);
  return State(std::move(succs[0].state));
}

inline PddlDeterministicDomain::Value
PddlDeterministicDomain::get_transition_value(const State &s, const Action &a,
                                              const State &ns) const {
  if (_total_cost_idx >= 0 &&
      _total_cost_idx < static_cast<int>(ns.fluents.size())) {
    auto &ns_map = ns.fluents[_total_cost_idx];
    auto &s_map = s.fluents[_total_cost_idx];
    GroundTuple empty_key;
    double ns_val = 0.0, s_val = 0.0;
    auto nit = ns_map.find(empty_key);
    if (nit != ns_map.end())
      ns_val = nit->second;
    auto sit = s_map.find(empty_key);
    if (sit != s_map.end())
      s_val = sit->second;
    return Value(ns_val - s_val);
  }
  return Value(1.0);
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_DOMAIN_ADAPTER_IMPL_HH
