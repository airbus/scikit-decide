/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_PREDICATE_FORMULA_HH
#define SKDECIDE_PDDL_PREDICATE_FORMULA_HH

#include "formula.hh"
#include "predicate.hh"
#include "term_container.hh"

namespace skdecide {

    namespace pddl {

        class PredicateFormula : public Formula,
                                 public TermContainer<PredicateFormula> {
        public :
            static constexpr char class_name[] = "predicate formula";

            typedef std::shared_ptr<PredicateFormula> Ptr;
            typedef TermContainer<PredicateFormula>::TermPtr TermPtr;
            typedef TermContainer<PredicateFormula>::TermVector TermVector;

            PredicateFormula() {}

            PredicateFormula(const Predicate::Ptr& predicate,
                             const TermContainer<PredicateFormula>& terms)
                : TermContainer<PredicateFormula>(terms), _predicate(predicate) {}
            
            PredicateFormula(const PredicateFormula& other)
                : TermContainer<PredicateFormula>(other), _predicate(other._predicate) {}
            
            PredicateFormula& operator= (const PredicateFormula& other) {
                dynamic_cast<TermContainer<PredicateFormula>&>(*this) = other;
                this->_predicate = other._predicate;
                return *this;
            }

            virtual ~PredicateFormula() {}

            void set_predicate(const Predicate::Ptr& predicate) {
                _predicate = predicate;
            }

            const Predicate::Ptr& get_predicate() const {
                return _predicate;
            }

            const std::string& get_name() const {
                return _predicate->get_name();
            }

            virtual std::ostream& print(std::ostream& o) const {
                return TermContainer<PredicateFormula>::print(o);
            }
        
        private :
            Predicate::Ptr _predicate;
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_PREDICATE_FORMULA_HH