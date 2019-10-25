/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_PREDICATE_EFFECT_HH
#define AIRLAPS_PDDL_PREDICATE_EFFECT_HH

#include "effect.hh"
#include "predicate.hh"
#include "term_container.hh"

namespace airlaps {

    namespace pddl {

        class PredicateEffect : public Effect,
                                public TermContainer<PredicateEffect> {
        public :
            static constexpr char class_name[] = "predicate effect";

            typedef std::shared_ptr<PredicateEffect> Ptr;
            typedef TermContainer<PredicateEffect>::TermPtr TermPtr;
            typedef TermContainer<PredicateEffect>::TermVector TermVector;

            PredicateEffect() {}

            PredicateEffect(const Predicate::Ptr& predicate,
                            const TermContainer<PredicateEffect>& terms)
                : TermContainer<PredicateEffect>(terms), _predicate(predicate) {}
            
            PredicateEffect(const PredicateEffect& other)
                : TermContainer<PredicateEffect>(other), _predicate(other._predicate) {}
            
            PredicateEffect& operator= (const PredicateEffect& other) {
                dynamic_cast<TermContainer<PredicateEffect>&>(*this) = other;
                this->_predicate = other._predicate;
                return *this;
            }

            virtual ~PredicateEffect() {}

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
                return TermContainer<PredicateEffect>::print(o);
            }
        
        private :
            Predicate::Ptr _predicate;
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_PREDICATE_EFFECT_HH