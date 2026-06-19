/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/formula.hh"

namespace skdecide {

namespace pddl {

Formula::~Formula() {}

std::string Formula::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

void Formula::collect_positive_atoms(const Task &, const Binding &,
                                     const AtomCallback &) const {}

std::ostream &operator<<(std::ostream &o, const Formula &f) {
  return f.print(o);
}

} // namespace pddl

} // namespace skdecide
