/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_CLASS_CONTAINER_HH
#define SKDECIDE_PDDL_CLASS_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

namespace pddl {

class Class;

template <typename Derived>
class ClassContainer : public AssociativeContainer<Derived, Class> {
public:
  typedef typename AssociativeContainer<Derived, Class>::SymbolPtr ClassPtr;
  typedef typename AssociativeContainer<Derived, Class>::SymbolSet ClassSet;

  ClassContainer(const ClassContainer &other);
  ClassContainer &operator=(const ClassContainer &other);
  virtual ~ClassContainer();

  template <typename T> const ClassPtr &add_class(const T &cls);
  template <typename T> void remove_class(const T &cls);
  template <typename T> const ClassPtr &get_class(const T &cls) const;
  const ClassSet &get_classes() const;

protected:
  ClassContainer();
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/class_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_CLASS_CONTAINER_HH
