/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_FUNCTION_CONTAINER_HH
#define SKDECIDE_PDDL_FUNCTION_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

namespace pddl {

class Function;

template <typename Derived>
class FunctionContainer : public AssociativeContainer<Derived, Function> {
public:
  typedef
      typename AssociativeContainer<Derived, Function>::SymbolPtr FunctionPtr;
  typedef
      typename AssociativeContainer<Derived, Function>::SymbolSet FunctionSet;

  FunctionContainer(const FunctionContainer &other);
  FunctionContainer &operator=(const FunctionContainer &other);
  virtual ~FunctionContainer();

  template <typename T> const FunctionPtr &add_function(const T &function);

  template <typename T> void remove_function(const T &function);

  template <typename T>
  const FunctionPtr &get_function(const T &function) const;

  const FunctionSet &get_functions() const;

protected:
  FunctionContainer();
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/function_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_FUNCTION_CONTAINER_HH
