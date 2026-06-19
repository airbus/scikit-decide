/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/expression.hh"

namespace skdecide {

namespace pddl {

Expression::~Expression() {}

std::string Expression::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

std::ostream &operator<<(std::ostream &o, const Expression &e) {
  return e.print(o);
}

} // namespace pddl

} // namespace skdecide
