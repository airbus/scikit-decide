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

namespace skdecide {

    namespace pddl {

        template <typename Derived, typename Symbol>
        class SequenceContainer {
        public :
            SequenceContainer(const SequenceContainer& other)
            : _container(other._container) {}

            SequenceContainer& operator=(const SequenceContainer& other) {
                this->_container = other._container;
                return *this;
            }
        
        protected :
            typedef std::shared_ptr<Symbol> SymbolPtr;
            typedef std::vector<SymbolPtr> SymbolVector;

            SymbolVector _container;

            SequenceContainer() {}

            /**
             * Appends a symbol to the container.
             */
            const SymbolPtr& append(const SymbolPtr& symbol) {
                _container.push_back(symbol);
                return _container.back();
            }

            /**
             * Appends a symbol to the container.
             */
            const SymbolPtr& append(const std::string& symbol) {
                return append(std::make_shared<Symbol>(symbol));
            }

            /**
             * Removes all symbols matching the given one from the container.
             */
            void remove(const std::string& symbol) {
                std::vector <typename SymbolVector::const_iterator> v;
                for (typename SymbolVector::const_iterator i = _container.begin(); i != _container.end(); ++i) {
                    if ((*i)->get_name() == symbol) {
                        v.push_back(i);
                    }
                }
                for (const auto& i : v) {
                    _container.erase(i);
                }
            }

            /**
             * Removes all symbols matching the given one from the container.
             */
            void remove(const SymbolPtr& symbol) {
                remove(symbol->get_name());
            }

            /**
             * Gets all the symbols matching the given one from the container
             */
            SymbolVector get(const std::string& symbol) const {
                SymbolVector v;
                for (typename SymbolVector::const_iterator i = _container.begin(); i != _container.end(); ++i) {
                    if ((*i)->get_name() == symbol) {
                        v.push_back(*i);
                    }
                }
                return v;
            }

            /**
             * Gets all the symbols matching the given one from the container
             */
            SymbolVector get(const SymbolPtr& symbol) const {
                return get(symbol->get_name());
            }

            /**
             * Gets a symbol from the container at a given index
             * Throws an exception if the given index exceeds the container size
             */
            const SymbolPtr& at(const std::size_t& index) const {
                if (index >= _container.size()) {
                    throw std::out_of_range("SKDECIDE exception: index " + std::to_string(index) +
                                            " exceeds the size of the vector of " + std::string(Symbol::class_name) + "s of " +
                                            std::string(Derived::class_name) + " '" + static_cast<const Derived*>(this)->get_name() + "'");
                } else {
                    return _container[index];
                }
            }

            const SymbolVector& get_container() const {
                return _container;
            }
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_SEQUENCE_CONTAINER_HH
