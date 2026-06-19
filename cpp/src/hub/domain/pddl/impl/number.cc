/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/number.hh"

namespace skdecide {

namespace pddl {

// === Number::Number implementation ===

bool Number::is_double() const { return _impl->is_double(); }

double Number::as_double() const { return _impl->as_double(); }

long Number::as_long() const { return _impl->as_long(); }

std::ostream &Number::print(std::ostream &o) const { return _impl->print(o); };

// === Number::ImplBase implementation ===

Number::ImplBase::~ImplBase() {}

// === Number printing operator ===

std::ostream &operator<<(std::ostream &o, const Number &n) {
  return n.print(o);
}

} // namespace pddl

} // namespace skdecide
