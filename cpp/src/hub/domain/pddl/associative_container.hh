/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_ASSOCIATIVE_CONTAINER_HH
#define AIRLAPS_PDDL_ASSOCIATIVE_CONTAINER_HH

#include <unordered_set>
#include <memory>
#include <string>
#include <sstream>

namespace airlaps {

    namespace pddl {

        struct SymbolHash {
            template <typename SymbolPtr>
            inline std::size_t operator()(const SymbolPtr& s) const {
                return std::hash<std::string>()(s->get_name());
            }
        };

        struct SymbolEqual {
            template <typename SymbolPtr>
            inline bool operator()(const SymbolPtr& s1, const SymbolPtr& s2) const {
                return std::equal_to<std::string>()(s1->get_name(), s2->get_name());
            }
        };

        template <typename Derived, typename Symbol>
        class AssociativeContainer {
        public :
            AssociativeContainer(const AssociativeContainer& other)
            : _container(other._container) {}

            AssociativeContainer& operator=(const AssociativeContainer& other) {
                this->_container = other._container;
                return *this;
            }
        
        protected :
            typedef std::shared_ptr<Symbol> SymbolPtr;
            typedef std::unordered_set<SymbolPtr, SymbolHash, SymbolEqual> SymbolSet;

            SymbolSet _container;

            AssociativeContainer() {}

            /**
             * Adds a symbol to the container.
             * Throws an exception if the given symbol is already in the symbol container
             */
            const SymbolPtr& add(const SymbolPtr& symbol) {
                std::pair<typename SymbolSet::const_iterator, bool> i = _container.emplace(symbol);
                if (!i.second) {
                    throw std::logic_error("AIRLAPS exception: " + std::string(Symbol::class_name) + " '" +
                                           symbol->get_name() +
                                           "' already in the set of " + std::string(Symbol::class_name) + "s of " +
                                           std::string(Derived::class_name) + " '" + static_cast<const Derived*>(this)->get_name() + "'");
                } else {
                    return *i.first;
                }
            }

            /**
             * Adds a symbol to the container.
             * Throws an exception if the given symbol is already in the symbol container
             */
            const SymbolPtr& add(const std::string& symbol) {
                return add(std::make_shared<Symbol>(symbol));
            }

            /**
             * Removes a symbol from the container.
             * Throws an exception if the given symbol is not in the symbol container
             */
            void remove(const SymbolPtr& symbol) {
                if (_container.erase(symbol) == 0) {
                    throw std::logic_error("AIRLAPS exception: " + std::string(Symbol::class_name) + " '" +
                                           symbol->get_name() +
                                           "' not in the set of " + std::string(Symbol::class_name) + "s of " +
                                           std::string(Derived::class_name) + " '" + static_cast<const Derived*>(this)->get_name() + "'");
                }
            }

            /**
             * Removes a symbol from the container.
             * Throws an exception if the given symbol is not in the symbol container
             */
            void remove(const std::string& symbol) {
                remove(std::make_shared<Symbol>(symbol));
            }

            /**
             * Gets a symbol from the container.
             * Throws an exception if the given symbol is not in the symbol container
             */
            const SymbolPtr& get(const SymbolPtr& symbol) const {
                typename SymbolSet::const_iterator i = _container.find(symbol);
                if (i == _container.end()) {
                    throw std::logic_error("AIRLAPS exception: " + std::string(Symbol::class_name) + " '" +
                                           symbol->get_name() +
                                           "' not in the set of " + std::string(Symbol::class_name) + "s of " +
                                           std::string(Derived::class_name) + " '" + static_cast<const Derived*>(this)->get_name() + "'");
                } else {
                    return *i;
                }
            }

             /**
             * Gets a symbol from the container.
             * Throws an exception if the given symbol is not in the symbol container
             */
            const SymbolPtr& get(const std::string& symbol) const {
                return get(std::make_shared<Symbol>(symbol));
            }

            const SymbolSet& get_container() const {
                return _container;
            }
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_ASSOCIATIVE_CONTAINER_HH
