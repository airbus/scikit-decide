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
        public :
            typedef typename AssociativeContainer<Derived, Function>::SymbolPtr FunctionPtr;
            typedef typename AssociativeContainer<Derived, Function>::SymbolSet FunctionSet;

            FunctionContainer(const FunctionContainer& other)
                : AssociativeContainer<Derived, Function>(other) {}
            
            FunctionContainer& operator=(const FunctionContainer& other) {
                dynamic_cast<AssociativeContainer<Derived, Function>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const FunctionPtr& add_function(const T& function) {
                return AssociativeContainer<Derived, Function>::add(function);
            }

            template <typename T>
            inline void remove_function(const T& function) {
                AssociativeContainer<Derived, Function>::remove(function);
            }

            template <typename T>
            inline const FunctionPtr& get_function(const T& function) const {
                return AssociativeContainer<Derived, Function>::get(function);
            }

            inline const FunctionSet& get_functions() const {
                return AssociativeContainer<Derived, Function>::get_container();
            }
        
        protected :
            FunctionContainer() {}
        };     

    } // namespace pddl
    
} // namespace skdecide

#endif // SKDECIDE_PDDL_FUNCTION_CONTAINER_HH
