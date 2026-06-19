/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/term.hh"

namespace skdecide {

namespace pddl {

std::ostream &operator<<(std::ostream &o, const Term &t) { return t.print(o); }

} // namespace pddl

} // namespace skdecide
