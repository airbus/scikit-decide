/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/function.hh"

namespace skdecide {

namespace pddl {

Function::Function(const std::string &name) : Identifier(name) {}

Function::Function(const Function &other)
    : Identifier(other), VariableContainer<Function>(other) {}

Function &Function::operator=(const Function &other) {
  dynamic_cast<Identifier &>(*this) = other;
  dynamic_cast<VariableContainer<Function> &>(*this) = other;
  return *this;
}

Function::~Function() {}

std::ostream &operator<<(std::ostream &o, const Function &f) {
  return f.print(o);
}

} // namespace pddl

} // namespace skdecide
