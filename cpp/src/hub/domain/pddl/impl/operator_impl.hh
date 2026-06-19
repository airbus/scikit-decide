/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/operator.hh"

namespace skdecide {

namespace pddl {

template <typename Derived>
Operator<Derived>::Operator(const std::string &name) : Identifier(name) {}

template <typename Derived>
Operator<Derived>::Operator(const std::string &name,
                            const Formula::Ptr &precondition,
                            const Effect::Ptr &effect)
    : Identifier(name), BinaryEffect(precondition, effect) {}

template <typename Derived>
Operator<Derived>::Operator(const Operator<Derived> &other)
    : Identifier(other), VariableContainer<Derived>(other),
      BinaryEffect(other) {}

template <typename Derived>
Operator<Derived> &
Operator<Derived>::operator=(const Operator<Derived> &other) {
  dynamic_cast<Identifier &>(*this) = other;
  dynamic_cast<VariableContainer<Derived> &>(*this) = other;
  dynamic_cast<BinaryEffect &>(*this) = other;
  return *this;
}

template <typename Derived> Operator<Derived>::~Operator() {}

template <typename Derived>
std::ostream &Operator<Derived>::print(std::ostream &o) const {
  o << "(:" << std::string(Derived::class_name) << ' '
    << static_cast<const Derived *>(this)->get_name() << std::endl;
  o << ":parameters (";
  for (const auto &v : this->get_variables()) {
    o << " " << *v;
  }
  o << " )" << std::endl;
  o << ":precondition " << *(this->get_condition()) << std::endl;
  o << ":effect " << *(this->get_effect()) << std::endl;
  o << ")";
  return o;
}

} // namespace pddl

} // namespace skdecide
