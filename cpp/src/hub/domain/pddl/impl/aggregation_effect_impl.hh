/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/aggregation_effect.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> AggregationEffect<Derived>::AggregationEffect() {}

template <typename Derived>
AggregationEffect<Derived>::AggregationEffect(
    const AggregationEffect<Derived> &other)
    : _effects(other._effects) {}

template <typename Derived>
AggregationEffect<Derived> &
AggregationEffect<Derived>::operator=(const AggregationEffect<Derived> &other) {
  this->_effects = other._effects;
  return *this;
}

template <typename Derived> AggregationEffect<Derived>::~AggregationEffect() {}

template <typename Derived>
AggregationEffect<Derived> &
AggregationEffect<Derived>::append_effect(const Effect::Ptr &effect) {
  _effects.push_back(effect);
  return *this;
}

template <typename Derived>
AggregationEffect<Derived> &
AggregationEffect<Derived>::remove_effect(const std::size_t &index) {
  if (index >= _effects.size()) {
    throw std::out_of_range(
        "SKDECIDE exception: index " + std::to_string(index) +
        " exceeds the size of the '" + Derived::class_name + "' effect");
  } else {
    _effects.erase(_effects.begin() + index);
    return *this;
  }
}

template <typename Derived>
const Effect::Ptr &
AggregationEffect<Derived>::effect_at(const std::size_t &index) {
  if (index >= _effects.size()) {
    throw std::out_of_range(
        "SKDECIDE exception: index " + std::to_string(index) +
        " exceeds the size of the '" + Derived::class_name + "' effect");
  } else {
    return _effects[index];
  }
}

template <typename Derived>
const typename AggregationEffect<Derived>::EffectVector &
AggregationEffect<Derived>::get_effects() const {
  return _effects;
}

template <typename Derived>
std::ostream &AggregationEffect<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name;
  for (const auto &f : _effects) {
    o << " " << *f;
  }
  o << ")";
  return o;
}

template <typename Derived>
void AggregationEffect<Derived>::collect_add_atoms(
    const Task &task, const Binding &binding,
    const AtomCallback &callback) const {
  for (auto &sub : get_effects()) {
    sub->collect_add_atoms(task, binding, callback);
  }
}

template <typename Derived>
void AggregationEffect<Derived>::collect_cost_increase(
    const Task &task, const Binding &binding,
    const CostCallback &callback) const {
  for (auto &sub : get_effects()) {
    sub->collect_cost_increase(task, binding, callback);
  }
}

} // namespace pddl

} // namespace skdecide
