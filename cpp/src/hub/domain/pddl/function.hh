/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FUNCTION_HH
#define SKDECIDE_PDDL_FUNCTION_HH

#include "identifier.hh"
#include "variable_container.hh"

namespace skdecide {

namespace pddl {

class Function : public Identifier, public VariableContainer<Function> {
public:
  static constexpr char class_name[] = "function";

  typedef std::shared_ptr<Function> Ptr;

  Function(const std::string &name);
  Function(const Function &other);
  Function &operator=(const Function &other);
  virtual ~Function();

  typedef VariableContainer<Function>::VariablePtr VariablePtr;
  typedef VariableContainer<Function>::VariableVector VariableVector;
};

// Object printing operator
std::ostream &operator<<(std::ostream &o, const Function &f);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_FUNCTION_HH
