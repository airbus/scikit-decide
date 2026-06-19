/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_TERM_CONTAINER_HH
#define SKDECIDE_PDDL_TERM_CONTAINER_HH

#include "sequence_container.hh"

namespace skdecide {

namespace pddl {

class Term;

template <typename Derived>
class TermContainer : public SequenceContainer<Derived, Term> {
public:
  typedef typename SequenceContainer<Derived, Term>::SymbolPtr TermPtr;
  typedef typename SequenceContainer<Derived, Term>::SymbolVector TermVector;

  TermContainer(const TermContainer &other)
      : SequenceContainer<Derived, Term>(other) {}

  TermContainer &operator=(const TermContainer &other);

  virtual ~TermContainer() {}

  const TermPtr &append_term(const TermPtr &term);

  template <typename T> void remove_term(const T &term) {
    SequenceContainer<Derived, Term>::remove(term);
  }

  template <typename T> TermVector get_term(const T &term) const {
    return SequenceContainer<Derived, Term>::get(term);
  }

  const TermPtr &term_at(const std::size_t &index) const;

  const TermVector &get_terms() const;

  std::ostream &print(std::ostream &o) const;

  std::string print() const;

protected:
  TermContainer() {}
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/term_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_TERM_CONTAINER_HH
