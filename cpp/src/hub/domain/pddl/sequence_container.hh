/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_SEQUENCE_CONTAINER_HH
#define SKDECIDE_PDDL_SEQUENCE_CONTAINER_HH

#include <vector>
#include <memory>
#include <string>
#include <sstream>
#include <algorithm>

#include "utils/string_converter.hh"

namespace skdecide {

namespace pddl {

class Variable;

template <typename Derived, typename Symbol> class SequenceContainer {
public:
  SequenceContainer(const SequenceContainer &other);
  SequenceContainer &operator=(const SequenceContainer &other);
  virtual ~SequenceContainer();

protected:
  typedef std::shared_ptr<Symbol> SymbolPtr;
  typedef std::vector<SymbolPtr> SymbolVector;

  SymbolVector _container;

  SequenceContainer();

  /**
   * Appends a symbol to the container.
   */
  const SymbolPtr &append(const SymbolPtr &symbol);

  /**
   * Appends a symbol to the container.
   */
  template <typename Tsymbol = Symbol,
            typename std::enable_if_t<std::is_same<Tsymbol, Variable>::value,
                                      int> = 0>
  const SymbolPtr &append(const std::string &symbol);

  /**
   * Removes all symbols matching the given one from the container.
   */
  void remove(const std::string &symbol);

  /**
   * Removes all symbols matching the given one from the container.
   */
  void remove(const SymbolPtr &symbol);

  /**
   * Gets all the symbols matching the given one from the container
   */
  SymbolVector get(const std::string &symbol) const;

  /**
   * Gets all the symbols matching the given one from the container
   */
  SymbolVector get(const SymbolPtr &symbol) const;

  /**
   * Gets a symbol from the container at a given index
   * Throws an exception if the given index exceeds the container size
   */
  const SymbolPtr &at(const std::size_t &index) const;

  const SymbolVector &get_container() const;
};

} // namespace pddl

} // namespace skdecide

#ifdef SKDECIDE_HEADERS_ONLY
#include "impl/sequence_container_impl.hh"
#endif

#endif // SKDECIDE_PDDL_SEQUENCE_CONTAINER_HH
