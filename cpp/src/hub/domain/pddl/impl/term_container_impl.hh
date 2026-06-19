/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/term_container.hh"

#include <sstream>

namespace skdecide {

namespace pddl {

template <typename Derived>
TermContainer<Derived> &
TermContainer<Derived>::operator=(const TermContainer &other) {
  dynamic_cast<SequenceContainer<Derived, Term> &>(*this) = other;
  return *this;
}

template <typename Derived>
const typename TermContainer<Derived>::TermPtr &
TermContainer<Derived>::append_term(const TermPtr &term) {
  return SequenceContainer<Derived, Term>::append(term);
}

template <typename Derived>
const typename TermContainer<Derived>::TermPtr &
TermContainer<Derived>::term_at(const std::size_t &index) const {
  return SequenceContainer<Derived, Term>::at(index);
}

template <typename Derived>
const typename TermContainer<Derived>::TermVector &
TermContainer<Derived>::get_terms() const {
  return SequenceContainer<Derived, Term>::get_container();
}

template <typename Derived>
std::ostream &TermContainer<Derived>::print(std::ostream &o) const {
  o << "(" << static_cast<const Derived *>(this)->get_name();
  for (const auto &t : get_terms()) {
    o << " " << t->get_name();
  }
  o << ")";
  return o;
}

template <typename Derived> std::string TermContainer<Derived>::print() const {
  std::ostringstream o;
  print(o);
  return o.str();
}

} // namespace pddl

} // namespace skdecide
