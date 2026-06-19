/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/unary_effect.hh"

namespace skdecide {

namespace pddl {

template <typename Derived, typename Eff>
UnaryEffect<Derived, Eff> &
UnaryEffect<Derived, Eff>::operator=(const UnaryEffect<Derived, Eff> &other) {
  this->_effect = other._effect;
  return *this;
}

template <typename Derived, typename Eff>
void UnaryEffect<Derived, Eff>::set_effect(const typename Eff::Ptr &effect) {
  _effect = effect;
}

template <typename Derived, typename Eff>
const typename Eff::Ptr &UnaryEffect<Derived, Eff>::get_effect() const {
  return _effect;
}

template <typename Derived, typename Eff>
std::ostream &UnaryEffect<Derived, Eff>::print(std::ostream &o) const {
  o << "(" << Derived::class_name << " " << *_effect << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
