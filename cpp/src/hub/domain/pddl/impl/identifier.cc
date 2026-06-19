/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/identifier.hh"

namespace skdecide {

namespace pddl {

Identifier::Identifier() {}

Identifier::Identifier(const Identifier &other) : _name(other._name) {}

Identifier::Identifier(const std::string &name)
    : _name(StringConverter::tolower(name)) {}

Identifier &Identifier::operator=(const Identifier &other) {
  this->_name = other._name;
  return *this;
}

Identifier::~Identifier() {}

void Identifier::set_name(const std::string &name) {
  _name = StringConverter::tolower(name);
}

const std::string &Identifier::get_name() const { return _name; }

} // namespace pddl

} // namespace skdecide
