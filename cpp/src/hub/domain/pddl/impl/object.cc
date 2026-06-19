/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/object.hh"

namespace skdecide {

namespace pddl {

Object::Object(const std::string &name) : Identifier(name) {}

Object::Object(const Object &other)
    : Identifier(other), TypeContainer<Object>(other) {}

Object &Object::operator=(const Object &other) {
  dynamic_cast<Identifier &>(*this) = other;
  dynamic_cast<TypeContainer<Object> &>(*this) = other;
  return *this;
}

Object::~Object() {}

const std::string &Object::get_name() const { return Identifier::get_name(); }

std::ostream &Object::print(std::ostream &o) const {
  return TypeContainer<Object>::print(o);
}

std::ostream &operator<<(std::ostream &o, const Object &ob) {
  return ob.print(o);
}

} // namespace pddl

} // namespace skdecide
