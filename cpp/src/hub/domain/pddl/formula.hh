/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FORMULA_HH
#define SKDECIDE_PDDL_FORMULA_HH

#include <functional>
#include <memory>
#include <ostream>
#include <sstream>

#include "semantics/fwd.hh"
#include "semantics/state.hh"

namespace skdecide {

namespace pddl {

class Formula {
public:
  typedef std::shared_ptr<Formula> Ptr;

  using AtomCallback = std::function<void(int, const GroundTuple &)>;

  virtual ~Formula();

  virtual std::ostream &print(std::ostream &o) const = 0;
  std::string print() const;

  virtual bool holds(const State &state, const Task &task,
                     const Binding &binding) const = 0;

  virtual void collect_positive_atoms(const Task &task, const Binding &binding,
                                      const AtomCallback &callback) const;
};

// Formula printing operator
std::ostream &operator<<(std::ostream &o, const Formula &f);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_FORMULA_HH
