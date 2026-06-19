/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_DETERMINIZED_DOMAIN_IMPL_HH
#define SKDECIDE_DETERMINIZED_DOMAIN_IMPL_HH

#include "hub/solver/determinization/determinized_domain.hh"

#include <algorithm>
#include <random>

namespace skdecide {

// === DeterminizedDomain implementation ===

#define SK_DET_DOMAIN_TEMPLATE_DECL                                            \
  template <typename Tdomain, typename TstrategyTag, typename Texec>

#define SK_DET_DOMAIN_CLASS DeterminizedDomain<Tdomain, TstrategyTag, Texec>

SK_DET_DOMAIN_TEMPLATE_DECL
SK_DET_DOMAIN_CLASS::DeterminizedDomain(Tdomain &domain) : _domain(domain) {
  if constexpr (std::is_same_v<TstrategyTag, RandomOutcomeStrategy>) {
    _rng.seed(std::random_device{}());
  }
}

SK_DET_DOMAIN_TEMPLATE_DECL
typename SK_DET_DOMAIN_CLASS::ActionSpace
SK_DET_DOMAIN_CLASS::get_applicable_actions(const State &s) {
  auto orig_actions = _domain.get_applicable_actions(s);
  ActionSpace result;
  if constexpr (std::is_same_v<TstrategyTag, AllOutcomesStrategy>) {
    const auto &elements = orig_actions.get_elements();
    std::for_each(Texec::policy, elements.begin(), elements.end(),
                  [this, &s, &result](const auto &a) {
                    auto dist =
                        _domain.get_next_state_distribution(s, a).get_values();
                    _execution_policy.protect([&result, &a, &dist]() {
                      std::size_t idx = 0;
                      for (auto dv : dist) {
                        (void)dv;
                        result._actions.push_back({a, idx++});
                      }
                      if (idx == 0) {
                        result._actions.push_back({a, 0});
                      }
                    });
                  });
  } else {
    for (const auto &a : orig_actions.get_elements()) {
      result._actions.push_back(a);
    }
  }
  return result;
}

SK_DET_DOMAIN_TEMPLATE_DECL
typename SK_DET_DOMAIN_CLASS::State
SK_DET_DOMAIN_CLASS::get_next_state(const State &s, const Action &a) {
  OrigAction orig_a;
  if constexpr (std::is_same_v<TstrategyTag, AllOutcomesStrategy>) {
    orig_a = a.original_action;
  } else {
    orig_a = a;
  }
  auto dist = _domain.get_next_state_distribution(s, orig_a).get_values();
  std::vector<double> weights;
  std::vector<State> states;
  for (auto ns : dist) {
    states.push_back(ns.state());
    weights.push_back(ns.probability());
  }
  if (states.empty()) {
    return s;
  }

  if constexpr (std::is_same_v<TstrategyTag, AllOutcomesStrategy>) {
    std::size_t idx = (a.outcome_index < states.size()) ? a.outcome_index : 0;
    return states[idx];
  } else if constexpr (std::is_same_v<TstrategyTag,
                                      MostProbableOutcomeStrategy>) {
    auto it = std::max_element(weights.begin(), weights.end());
    return states[std::distance(weights.begin(), it)];
  } else {
    std::discrete_distribution<> d(weights.begin(), weights.end());
    return states[d(_rng)];
  }
}

SK_DET_DOMAIN_TEMPLATE_DECL
typename SK_DET_DOMAIN_CLASS::Value
SK_DET_DOMAIN_CLASS::get_transition_value(const State &s, const Action &a,
                                          const State &ns) {
  OrigAction orig_a;
  if constexpr (std::is_same_v<TstrategyTag, AllOutcomesStrategy>) {
    orig_a = a.original_action;
  } else {
    orig_a = a;
  }
  return _domain.get_transition_value(s, orig_a, ns);
}

// === TransitionDeterminizationAdapter implementation ===

#define SK_TRANS_ADAPTER_TEMPLATE_DECL                                         \
  template <typename Tdomain, typename TstrategyTag, typename Texec>

#define SK_TRANS_ADAPTER_CLASS                                                 \
  TransitionDeterminizationAdapter<Tdomain, TstrategyTag, Texec>

SK_TRANS_ADAPTER_TEMPLATE_DECL
SK_TRANS_ADAPTER_CLASS::TransitionDeterminizationAdapter(Tdomain &domain)
    : _domain(domain) {}

SK_TRANS_ADAPTER_TEMPLATE_DECL
void SK_TRANS_ADAPTER_CLASS::update() {
  _det_domain = std::make_unique<DeterminizedDomainType>(_domain);
}

SK_TRANS_ADAPTER_TEMPLATE_DECL
typename SK_TRANS_ADAPTER_CLASS::DeterminizedDomainType &
SK_TRANS_ADAPTER_CLASS::domain() {
  return *_det_domain;
}

SK_TRANS_ADAPTER_TEMPLATE_DECL
typename SK_TRANS_ADAPTER_CLASS::Action
SK_TRANS_ADAPTER_CLASS::to_original(const DetAction &a) const {
  if constexpr (std::is_same_v<TstrategyTag, AllOutcomesStrategy>) {
    return a.original_action;
  } else {
    return a;
  }
}

SK_TRANS_ADAPTER_TEMPLATE_DECL
typename SK_TRANS_ADAPTER_CLASS::State
SK_TRANS_ADAPTER_CLASS::expected_next(const State &s, const DetAction &a) {
  return _det_domain->get_next_state(s, a);
}

SK_TRANS_ADAPTER_TEMPLATE_DECL
bool SK_TRANS_ADAPTER_CLASS::needs_update_each_replan() const {
  return std::is_same_v<TstrategyTag, RandomOutcomeStrategy>;
}

} // namespace skdecide

#endif // SKDECIDE_DETERMINIZED_DOMAIN_IMPL_HH
