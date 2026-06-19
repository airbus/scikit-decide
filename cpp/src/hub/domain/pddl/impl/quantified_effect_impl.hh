/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/quantified_effect.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> QuantifiedEffect<Derived>::QuantifiedEffect() {}

template <typename Derived>
QuantifiedEffect<Derived>::QuantifiedEffect(
    const Effect::Ptr &effect, const VariableContainer<Derived> &variables)
    : VariableContainer<Derived>(variables), _effect(effect) {}

template <typename Derived>
QuantifiedEffect<Derived>::QuantifiedEffect(
    const QuantifiedEffect<Derived> &other)
    : VariableContainer<Derived>(other), _effect(other._effect) {}

template <typename Derived>
QuantifiedEffect<Derived> &
QuantifiedEffect<Derived>::operator=(const QuantifiedEffect<Derived> &other) {
  dynamic_cast<VariableContainer<Derived> &>(*this) = other;
  this->_effect = other._effect;
  return *this;
}

template <typename Derived> QuantifiedEffect<Derived>::~QuantifiedEffect() {}

template <typename Derived>
QuantifiedEffect<Derived> &
QuantifiedEffect<Derived>::set_effect(const Effect::Ptr &effect) {
  _effect = effect;
  return *this;
}

template <typename Derived>
const Effect::Ptr &QuantifiedEffect<Derived>::get_effect() const {
  return _effect;
}

template <typename Derived> const char *QuantifiedEffect<Derived>::get_name() {
  return Derived::class_name;
}

template <typename Derived>
std::ostream &QuantifiedEffect<Derived>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " (";
  for (const auto &v : this->get_variables()) {
    o << " " << *v;
  }
  o << ") " << *_effect << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
