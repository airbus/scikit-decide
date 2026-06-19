/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OBJECT_HH
#define SKDECIDE_PDDL_OBJECT_HH

#include "type.hh"
#include "term.hh"

namespace skdecide {

namespace pddl {

class Object : public Term, public Identifier, public TypeContainer<Object> {
public:
  static constexpr char class_name[] = "object";

  Object(const std::string &name);
  Object(const Object &other);
  Object &operator=(const Object &other);
  virtual ~Object();

  virtual const std::string &get_name() const;

  virtual std::ostream &print(std::ostream &o) const;
};

// Object printing operator
std::ostream &operator<<(std::ostream &o, const Object &ob);

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_OBJECT_HH
