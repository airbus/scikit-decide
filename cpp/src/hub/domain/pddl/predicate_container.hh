/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PREDICATE_CONTAINER_HH
#define SKDECIDE_PDDL_PREDICATE_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

namespace pddl {

class Predicate;

template <typename Derived>
class PredicateContainer : public AssociativeContainer<Derived, Predicate> {
public:
  typedef
      typename AssociativeContainer<Derived, Predicate>::SymbolPtr PredicatePtr;
  typedef
      typename AssociativeContainer<Derived, Predicate>::SymbolSet PredicateSet;

  PredicateContainer(const PredicateContainer &other);
  PredicateContainer &operator=(const PredicateContainer &other);
  virtual ~PredicateContainer();

  template <typename T> const PredicatePtr &add_predicate(const T &predicate);
  template <typename T> void remove_predicate(const T &predicate);

  template <typename T>
  const PredicatePtr &get_predicate(const T &predicate) const;

  const PredicateSet &get_predicates() const;

protected:
  PredicateContainer();
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/predicate_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_PREDICATE_CONTAINER_HH
