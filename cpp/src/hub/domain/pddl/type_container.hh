/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_TYPE_CONTAINER_HH
#define SKDECIDE_PDDL_TYPE_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

namespace pddl {

class Type;

template <typename Derived>
class TypeContainer : public AssociativeContainer<Derived, Type> {
public:
  typedef typename AssociativeContainer<Derived, Type>::SymbolPtr TypePtr;
  typedef typename AssociativeContainer<Derived, Type>::SymbolSet TypeSet;

  TypeContainer(const TypeContainer &other)
      : AssociativeContainer<Derived, Type>(other) {}

  TypeContainer &operator=(const TypeContainer &other);

  template <typename T> const TypePtr &add_type(const T &type) {
    return AssociativeContainer<Derived, Type>::add(type);
  }

  template <typename T> void remove_type(const T &type) {
    AssociativeContainer<Derived, Type>::remove(type);
  }

  template <typename T> const TypePtr &get_type(const T &type) const {
    return AssociativeContainer<Derived, Type>::get(type);
  }

  const TypeSet &get_types() const;

  std::ostream &print(std::ostream &o) const;

  std::string print() const;

protected:
  TypeContainer() {}
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/type_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_TYPE_CONTAINER_HH
