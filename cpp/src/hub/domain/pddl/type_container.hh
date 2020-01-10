/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_TYPE_CONTAINER_HH
#define SKDECIDE_PDDL_TYPE_CONTAINER_HH

#include "associative_container.hh"

namespace skdecide {

    namespace pddl{

        class Type;

        template <typename Derived>
        class TypeContainer : public AssociativeContainer<Derived, Type> {
        public :
            typedef typename AssociativeContainer<Derived, Type>::SymbolPtr TypePtr;
            typedef typename AssociativeContainer<Derived, Type>::SymbolSet TypeSet;

            TypeContainer(const TypeContainer& other)
                : AssociativeContainer<Derived, Type>(other) {}
            
            TypeContainer& operator=(const TypeContainer& other) {
                dynamic_cast<AssociativeContainer<Derived, Type>&>(*this) = other;
                return *this;
            }

            template <typename T>
            inline const TypePtr& add_type(const T& type) {
                return AssociativeContainer<Derived, Type>::add(type);
            }

            template <typename T>
            inline void remove_type(const T& type) {
                AssociativeContainer<Derived, Type>::remove(type);
            }

            template <typename T>
            inline const TypePtr& get_type(const T& type) const {
                return AssociativeContainer<Derived, Type>::get(type);
            }

            inline const TypeSet& get_types() const {
                return AssociativeContainer<Derived, Type>::get_container();
            }

            std::ostream& print(std::ostream& o) const {
                o << static_cast<const Derived*>(this)->get_name();
                if (!get_types().empty()) {
                    o << " - ";
                    if (get_types().size() > 1) {
                        o << "(either";
                        for (const auto& t : get_types()) {
                            o << " " << t->get_name();
                        }
                        o << ")";
                    } else {
                        o << (*get_types().begin())->get_name();
                    }
                }
                return o;
            }

            std::string print() const {
                std::ostringstream o;
                print(o);
                return o.str();
            }
        
        protected :
            TypeContainer() {}
        };     

    } // namespace pddl
    
} // namespace airlap 

#endif // SKDECIDE_PDDL_TYPE_CONTAINER_HH
