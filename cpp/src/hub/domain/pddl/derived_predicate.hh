/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_DERIVED_PREDICATE_HH
#define AIRLAPS_PDDL_DERIVED_PREDICATE_HH

#include "predicate.hh"
#include "formula.hh"

namespace airlaps {

    namespace pddl {

        class DerivedPredicate {
        public :
            static constexpr char class_name[] = "derived predicate";

            typedef std::shared_ptr<DerivedPredicate> Ptr;

            DerivedPredicate(const std::string& name) {
                _predicate = std::make_shared<Predicate>(name);
            }

            DerivedPredicate(const Predicate::Ptr& predicate,
                             const Formula::Ptr& formula)
                : _predicate(predicate), _formula(formula) {}
            
            DerivedPredicate(const DerivedPredicate& other)
                : _predicate(other._predicate),
                  _formula(other._formula) {}
            
            DerivedPredicate& operator= (const DerivedPredicate& other) {
                this->_predicate = other._predicate;
                this->_formula = other._formula;
                return *this;
            }

            void set_predicate(const Predicate::Ptr& predicate) {
                _predicate = predicate;
            }

            const Predicate::Ptr& get_predicate() const {
                return _predicate;
            }

            void set_formula(const Formula::Ptr& formula) {
                _formula = formula;
            }

            const Formula::Ptr& get_formula() const {
                return _formula;
            }

            const std::string& get_name() const {
                return _predicate->get_name();
            }

            std::ostream& print(std::ostream& o) const {
                o << "(:derived " << *_predicate << " " << *_formula << ")";
                return o;
            }

        private :
            Predicate::Ptr _predicate;
            Formula::Ptr _formula;
        };

    } // namespace pddl

} // namespace airlaps

// Derived predicate printing operator
inline std::ostream& operator<<(std::ostream& o, const airlaps::pddl::DerivedPredicate& d) {
    return d.print(o);
}

#endif // AIRLAPS_PDDL_DERIVED_PREDICATE_HH
