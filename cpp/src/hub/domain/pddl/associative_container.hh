/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_ASSOCIATIVE_CONTAINER_HH
#define SKDECIDE_PDDL_ASSOCIATIVE_CONTAINER_HH

#include <unordered_set>
#include <memory>
#include <string>
#include <sstream>

namespace skdecide {

namespace pddl {

struct SymbolHash {
  template <typename SymbolPtr>
  std::size_t operator()(const SymbolPtr &s) const;
};

struct SymbolEqual {
  template <typename SymbolPtr>
  bool operator()(const SymbolPtr &s1, const SymbolPtr &s2) const;
};

template <typename Derived, typename Symbol> class AssociativeContainer {
public:
  AssociativeContainer(const AssociativeContainer &other);
  AssociativeContainer &operator=(const AssociativeContainer &other);
  virtual ~AssociativeContainer();

protected:
  typedef std::shared_ptr<Symbol> SymbolPtr;
  typedef std::unordered_set<SymbolPtr, SymbolHash, SymbolEqual> SymbolSet;

  SymbolSet _container;

  AssociativeContainer();

  /**
   * Adds a symbol to the container.
   * Throws an exception if the given symbol is already in the symbol container
   */
  const SymbolPtr &add(const SymbolPtr &symbol);

  /**
   * Adds a symbol to the container.
   * Throws an exception if the given symbol is already in the symbol container
   */
  const SymbolPtr &add(const std::string &symbol);

  /**
   * Removes a symbol from the container.
   * Throws an exception if the given symbol is not in the symbol container
   */
  void remove(const SymbolPtr &symbol);

  /**
   * Removes a symbol from the container.
   * Throws an exception if the given symbol is not in the symbol container
   */
  void remove(const std::string &symbol);

  /**
   * Gets a symbol from the container.
   * Throws an exception if the given symbol is not in the symbol container
   */
  const SymbolPtr &get(const SymbolPtr &symbol) const;

  /**
   * Gets a symbol from the container.
   * Throws an exception if the given symbol is not in the symbol container
   */
  const SymbolPtr &get(const std::string &symbol) const;

  const SymbolSet &get_container() const;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/associative_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_ASSOCIATIVE_CONTAINER_HH
