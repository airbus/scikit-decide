/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef SKDECIDE_PDDL_ASSOCIATIVE_CONTAINER_IMPL_HH
#define SKDECIDE_PDDL_ASSOCIATIVE_CONTAINER_IMPL_HH

#include "hub/domain/pddl/associative_container.hh"

namespace skdecide {

namespace pddl {

// === SymbolHash / SymbolEqual implementation ===

template <typename SymbolPtr>
std::size_t SymbolHash::operator()(const SymbolPtr &s) const {
  return std::hash<std::string>()(s->get_name());
}

template <typename SymbolPtr>
bool SymbolEqual::operator()(const SymbolPtr &s1, const SymbolPtr &s2) const {
  return std::equal_to<std::string>()(s1->get_name(), s2->get_name());
}

// === AssociativeContainer implementation ===

template <typename Derived, typename Symbol>
AssociativeContainer<Derived, Symbol>::AssociativeContainer(
    const AssociativeContainer &other)
    : _container(other._container) {}

template <typename Derived, typename Symbol>
AssociativeContainer<Derived, Symbol> &
AssociativeContainer<Derived, Symbol>::operator=(
    const AssociativeContainer<Derived, Symbol> &other) {
  this->_container = other._container;
  return *this;
}

template <typename Derived, typename Symbol>
AssociativeContainer<Derived, Symbol>::AssociativeContainer() {}

template <typename Derived, typename Symbol>
AssociativeContainer<Derived, Symbol>::~AssociativeContainer() {}

template <typename Derived, typename Symbol>
const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &
AssociativeContainer<Derived, Symbol>::add(
    const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &symbol) {
  std::pair<typename SymbolSet::const_iterator, bool> i =
      _container.emplace(symbol);
  if (!i.second) {
    throw std::logic_error(
        "SKDECIDE exception: " + std::string(Symbol::class_name) + " '" +
        symbol->get_name() + "' already in the set of " +
        std::string(Symbol::class_name) + "s of " +
        std::string(Derived::class_name) + " '" +
        static_cast<const Derived *>(this)->get_name() + "'");
  } else {
    return *i.first;
  }
}

template <typename Derived, typename Symbol>
const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &
AssociativeContainer<Derived, Symbol>::add(const std::string &symbol) {
  return add(std::make_shared<Symbol>(symbol));
}

template <typename Derived, typename Symbol>
void AssociativeContainer<Derived, Symbol>::remove(
    const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &symbol) {
  if (_container.erase(symbol) == 0) {
    throw std::logic_error(
        "SKDECIDE exception: " + std::string(Symbol::class_name) + " '" +
        symbol->get_name() + "' not in the set of " +
        std::string(Symbol::class_name) + "s of " +
        std::string(Derived::class_name) + " '" +
        static_cast<const Derived *>(this)->get_name() + "'");
  }
}

template <typename Derived, typename Symbol>
void AssociativeContainer<Derived, Symbol>::remove(const std::string &symbol) {
  remove(std::make_shared<Symbol>(symbol));
}

template <typename Derived, typename Symbol>
const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &
AssociativeContainer<Derived, Symbol>::get(
    const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &symbol)
    const {
  typename SymbolSet::const_iterator i = _container.find(symbol);
  if (i == _container.end()) {
    throw std::logic_error(
        "SKDECIDE exception: " + std::string(Symbol::class_name) + " '" +
        symbol->get_name() + "' not in the set of " +
        std::string(Symbol::class_name) + "s of " +
        std::string(Derived::class_name) + " '" +
        static_cast<const Derived *>(this)->get_name() + "'");
  } else {
    return *i;
  }
}

template <typename Derived, typename Symbol>
const typename AssociativeContainer<Derived, Symbol>::SymbolPtr &
AssociativeContainer<Derived, Symbol>::get(const std::string &symbol) const {
  return get(std::make_shared<Symbol>(symbol));
}

template <typename Derived, typename Symbol>
const typename AssociativeContainer<Derived, Symbol>::SymbolSet &
AssociativeContainer<Derived, Symbol>::get_container() const {
  return _container;
}

} // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_ASSOCIATIVE_CONTAINER_IMPL_HH
