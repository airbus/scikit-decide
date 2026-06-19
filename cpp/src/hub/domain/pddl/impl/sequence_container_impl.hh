/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "hub/domain/pddl/sequence_container.hh"

namespace skdecide {

namespace pddl {

template <typename Derived, typename Symbol>
SequenceContainer<Derived, Symbol>::SequenceContainer() {}

template <typename Derived, typename Symbol>
SequenceContainer<Derived, Symbol>::SequenceContainer(
    const SequenceContainer<Derived, Symbol> &other)
    : _container(other._container) {}

template <typename Derived, typename Symbol>
SequenceContainer<Derived, Symbol> &
SequenceContainer<Derived, Symbol>::operator=(const SequenceContainer &other) {
  this->_container = other._container;
  return *this;
}

template <typename Derived, typename Symbol>
SequenceContainer<Derived, Symbol>::~SequenceContainer() {}

template <typename Derived, typename Symbol>
const typename SequenceContainer<Derived, Symbol>::SymbolPtr &
SequenceContainer<Derived, Symbol>::append(const SymbolPtr &symbol) {
  _container.push_back(symbol);
  return _container.back();
}

template <typename Derived, typename Symbol>
template <typename Tsymbol, typename std::enable_if_t<
                                std::is_same<Tsymbol, Variable>::value, int>>
const typename SequenceContainer<Derived, Symbol>::SymbolPtr &
SequenceContainer<Derived, Symbol>::append(const std::string &symbol) {
  return append(std::make_shared<Symbol>(symbol));
}

template <typename Derived, typename Symbol>
void SequenceContainer<Derived, Symbol>::remove(const std::string &symbol) {
  std::string lsymbol = StringConverter::tolower(symbol);
  std::vector<typename SymbolVector::const_iterator> v;
  for (typename SymbolVector::const_iterator i = _container.begin();
       i != _container.end(); ++i) {
    if ((*i)->get_name() == lsymbol) {
      v.push_back(i);
    }
  }
  for (const auto &i : v) {
    _container.erase(i);
  }
}

template <typename Derived, typename Symbol>
void SequenceContainer<Derived, Symbol>::remove(const SymbolPtr &symbol) {
  remove(symbol->get_name());
}

template <typename Derived, typename Symbol>
typename SequenceContainer<Derived, Symbol>::SymbolVector
SequenceContainer<Derived, Symbol>::get(const std::string &symbol) const {
  std::string lsymbol = StringConverter::tolower(symbol);
  SymbolVector v;
  for (typename SymbolVector::const_iterator i = _container.begin();
       i != _container.end(); ++i) {
    if ((*i)->get_name() == lsymbol) {
      v.push_back(*i);
    }
  }
  return v;
}

template <typename Derived, typename Symbol>
typename SequenceContainer<Derived, Symbol>::SymbolVector
SequenceContainer<Derived, Symbol>::get(const SymbolPtr &symbol) const {
  return get(symbol->get_name());
}

template <typename Derived, typename Symbol>
const typename SequenceContainer<Derived, Symbol>::SymbolPtr &
SequenceContainer<Derived, Symbol>::at(const std::size_t &index) const {
  if (index >= _container.size()) {
    throw std::out_of_range(
        "SKDECIDE exception: index " + std::to_string(index) +
        " exceeds the size of the vector of " +
        std::string(Symbol::class_name) + "s of " +
        std::string(Derived::class_name) + " '" +
        static_cast<const Derived *>(this)->get_name() + "'");
  } else {
    return _container[index];
  }
}

template <typename Derived, typename Symbol>
const typename SequenceContainer<Derived, Symbol>::SymbolVector &
SequenceContainer<Derived, Symbol>::get_container() const {
  return _container;
}

} // namespace pddl

} // namespace skdecide
