/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/number.hh"

namespace skdecide {

namespace pddl {

// === Number::Number implementation ===

template <typename T> Number::Number(const T &n) {
  _impl = std::make_unique<Impl<T>>(n);
}

// === Number::Impl implementation ===

template <typename T> Number::Impl<T>::Impl(const T &n) : _n(n) {}

template <typename T> Number::Impl<T>::~Impl() {}

template <typename T> bool Number::Impl<T>::is_double() const {
  return std::is_floating_point<T>::value;
}

template <typename T> double Number::Impl<T>::as_double() const {
  return (double)_n;
}

template <typename T> long Number::Impl<T>::as_long() const { return (long)_n; }

template <typename T>
std::ostream &Number::Impl<T>::print(std::ostream &o) const {
  o << _n;
  return o;
}

} // namespace pddl

} // namespace skdecide
