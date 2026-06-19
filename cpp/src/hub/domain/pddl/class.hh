/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_CLASS_HH
#define SKDECIDE_PDDL_CLASS_HH

#include "identifier.hh"
#include "function_container.hh"

namespace skdecide {

namespace pddl {

class Class : public Identifier, public FunctionContainer<Class> {
public:
  static constexpr char class_name[] = "class";

  typedef std::shared_ptr<Class> Ptr;
  typedef FunctionContainer<Class>::FunctionPtr FunctionPtr;
  typedef FunctionContainer<Class>::FunctionSet FunctionSet;

  Class(const std::string &name);
  Class(const Class &other);
  Class &operator=(const Class &other);
  virtual ~Class();

  std::ostream &print(std::ostream &o) const;
};

// Class printing operator
std::ostream &operator<<(std::ostream &o, const Class &c);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_CLASS_HH
