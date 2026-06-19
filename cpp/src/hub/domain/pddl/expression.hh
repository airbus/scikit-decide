/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_EXPRESSION_HH
#define SKDECIDE_PDDL_EXPRESSION_HH

#include <memory>
#include <ostream>
#include <sstream>

#include "semantics/fwd.hh"
#include "semantics/state.hh"

namespace skdecide {

namespace pddl {

class Expression {
public:
  typedef std::shared_ptr<Expression> Ptr;

  virtual ~Expression();

  virtual std::ostream &print(std::ostream &o) const = 0;
  std::string print() const;

  virtual double evaluate(const State &state, const Task &task,
                          const Binding &binding) const = 0;
};

// Expression printing operator
std::ostream &operator<<(std::ostream &o, const Expression &e);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_EXPRESSION_HH
