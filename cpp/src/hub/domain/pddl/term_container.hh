/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_TERM_CONTAINER_HH
#define AIRLAPS_PDDL_TERM_CONTAINER_HH

#include "sequence_container.hh"

namespace airlaps {

    namespace pddl {

        class Term;

        template <typename Derived>
        class TermContainer : public SequenceContainer<Derived, Term> {
        public :
            typedef typename SequenceContainer<Derived, Term>::SymbolPtr TermPtr;
            typedef typename SequenceContainer<Derived, Term>::SymbolVector TermVector;

            TermContainer(const TermContainer& other)
                : SequenceContainer<Derived, Term>(other) {}
            
            TermContainer& operator=(const TermContainer& other) {
                dynamic_cast<SequenceContainer<Derived, Term>&>(*this) = other;
                return *this;
            }

            // Terms cannot be created (we can only  create objects and
            // variables) so we just allow for passing shared pointers to terms
            inline const TermPtr& append_term(const TermPtr& term) {
                return SequenceContainer<Derived, Term>::append(term);
            }

            template <typename T>
            inline void remove_term(const T& term) {
                SequenceContainer<Derived, Term>::remove(term);
            }

            template <typename T>
            inline TermVector get_term(const T& term) const {
                return SequenceContainer<Derived, Term>::get(term);
            }

            inline const TermPtr& term_at(const std::size_t& index) const {
                return SequenceContainer<Derived, Term>::at(index);
            }

            inline const TermVector& get_terms() const {
                return SequenceContainer<Derived, Term>::get_container();
            }

            std::ostream& print(std::ostream& o) const {
                o << "(" << static_cast<const Derived*>(this)->get_name();
                for (const auto & t : get_terms()) {
                    o << " " << *t;
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
            TermContainer() {}
        };     

    } // namespace pddl
    
} // namespace airlaps

#endif // AIRLAPS_PDDL_TERM_CONTAINER_HH
