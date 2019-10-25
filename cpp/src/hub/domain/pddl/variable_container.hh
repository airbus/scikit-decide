/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_VARIABLE_CONTAINER_HH
#define AIRLAPS_PDDL_VARIABLE_CONTAINER_HH

#include "sequence_container.hh"

namespace airlaps {

    namespace pddl {

        class Variable;

        template <typename Derived>
        class VariableContainer : public SequenceContainer<Derived, Variable> {
        public :
            typedef typename SequenceContainer<Derived, Variable>::SymbolPtr VariablePtr;
            typedef typename SequenceContainer<Derived, Variable>::SymbolVector VariableVector;

            VariableContainer(const VariableContainer& other)
                : SequenceContainer<Derived, Variable>(other) {}
            
            VariableContainer& operator=(const VariableContainer& other) {
                dynamic_cast<SequenceContainer<Derived, Variable>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const VariablePtr& append_variable(const T& variable) {
                return SequenceContainer<Derived, Variable>::append(variable);
            }

            template <typename T>
            inline void remove_variable(const T& variable) {
                SequenceContainer<Derived, Variable>::remove(variable);
            }

            template <typename T>
            inline VariableVector get_variable(const T& variable) const {
                return SequenceContainer<Derived, Variable>::get(variable);
            }

            inline const VariablePtr& variable_at(const std::size_t& index) const {
                return SequenceContainer<Derived, Variable>::at(index);
            }

            inline const VariableVector& get_variables() const {
                return SequenceContainer<Derived, Variable>::get_container();
            }

            std::ostream& print(std::ostream& o) const {
                o << "(" << static_cast<const Derived*>(this)->get_name();
                for (const auto & v : get_variables()) {
                    o << " " << *v;
                }
                o << ")";
                return o;
            }

            std::string print() const {
                std::ostringstream o;
                print(o);
                return o.str();
            }
        
        protected :
            VariableContainer() {}
        };     

    } // namespace pddl
    
} // namespace airlaps

#endif // AIRLAPS_PDDL_VARIABLE_CONTAINER_HH
