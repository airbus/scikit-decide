/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/class.hh"
#include "hub/domain/pddl/function.hh"

namespace skdecide {

namespace pddl {

Class::Class(const std::string &name) : Identifier(name) {}

Class::Class(const Class &other)
    : Identifier(other), FunctionContainer<Class>(other) {}

Class &Class::operator=(const Class &other) {
  dynamic_cast<Identifier &>(*this) = other;
  dynamic_cast<FunctionContainer<Class> &>(*this) = other;
  return *this;
}

Class::~Class() {}

std::ostream &Class::print(std::ostream &o) const {
  o << "(:class" << this->_name;
  for (const auto &f : this->_container) {
    o << " " << *f;
  }
  o << ")";
  return o;
}

std::ostream &operator<<(std::ostream &o, const Class &c) { return c.print(o); }

} // namespace pddl

} // namespace skdecide
