/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/function_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived> FunctionContainer<Derived>::FunctionContainer() {}

template <typename Derived>
FunctionContainer<Derived>::FunctionContainer(
    const FunctionContainer<Derived> &other)
    : AssociativeContainer<Derived, Function>(other) {}

template <typename Derived>
FunctionContainer<Derived> &
FunctionContainer<Derived>::operator=(const FunctionContainer<Derived> &other) {
  dynamic_cast<AssociativeContainer<Derived, Function> &>(*this) = other;
  return *this;
}

template <typename Derived> FunctionContainer<Derived>::~FunctionContainer() {}

template <typename Derived>
template <typename T>
const typename FunctionContainer<Derived>::FunctionPtr &
FunctionContainer<Derived>::add_function(const T &function) {
  return AssociativeContainer<Derived, Function>::add(function);
}

template <typename Derived>
template <typename T>
void FunctionContainer<Derived>::remove_function(const T &function) {
  AssociativeContainer<Derived, Function>::remove(function);
}

template <typename Derived>
template <typename T>
const typename FunctionContainer<Derived>::FunctionPtr &
FunctionContainer<Derived>::get_function(const T &function) const {
  return AssociativeContainer<Derived, Function>::get(function);
}

template <typename Derived>
const typename FunctionContainer<Derived>::FunctionSet &
FunctionContainer<Derived>::get_functions() const {
  return AssociativeContainer<Derived, Function>::get_container();
}

} // namespace pddl

} // namespace skdecide
