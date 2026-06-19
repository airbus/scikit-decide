/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_DERIVED_PREDICATE_CONTAINER_HH
#define SKDECIDE_PDDL_DERIVED_PREDICATE_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

namespace pddl {

class DerivedPredicate;

template <typename Derived>
class DerivedPredicateContainer
    : public AssociativeContainer<Derived, DerivedPredicate> {
public:
  typedef typename AssociativeContainer<Derived, DerivedPredicate>::SymbolPtr
      DerivedPredicatePtr;
  typedef typename AssociativeContainer<Derived, DerivedPredicate>::SymbolSet
      DerivedPredicateSet;

  DerivedPredicateContainer(const DerivedPredicateContainer &other);
  DerivedPredicateContainer &operator=(const DerivedPredicateContainer &other);
  virtual ~DerivedPredicateContainer();

  template <typename T>
  const DerivedPredicatePtr &add_derived_predicate(const T &derived_predicate);

  template <typename T>
  void remove_derived_predicate(const T &derived_predicate);

  template <typename T>
  const DerivedPredicatePtr &
  get_derived_predicate(const T &derived_predicate) const;

  const DerivedPredicateSet &get_derived_predicates() const;

protected:
  DerivedPredicateContainer();
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/derived_predicate_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_DERIVED_PREDICATE_CONTAINER_HH
