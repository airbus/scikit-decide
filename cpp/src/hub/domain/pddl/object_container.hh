/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_OBJECT_CONTAINER_HH
#define SKDECIDE_PDDL_OBJECT_CONTAINER_HH

#include "associative_container.hh"
#include "object.hh"

namespace skdecide {

namespace pddl {

class Object;

template <typename Derived>
class ObjectContainer : public AssociativeContainer<Derived, Object> {
public:
  typedef typename AssociativeContainer<Derived, Object>::SymbolPtr ObjectPtr;
  typedef typename AssociativeContainer<Derived, Object>::SymbolSet ObjectSet;

  ObjectContainer(const ObjectContainer &other);
  ObjectContainer &operator=(const ObjectContainer &other);
  virtual ~ObjectContainer();

  template <typename T> const ObjectPtr &add_object(const T &object);
  template <typename T> void remove_object(const T &object);

  template <typename T> const ObjectPtr &get_object(const T &object) const;

  const ObjectSet &get_objects() const;

  virtual std::ostream &print(std::ostream &o) const;
  virtual std::string print() const;

protected:
  ObjectContainer();
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/object_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_OBJECT_CONTAINER_HH
