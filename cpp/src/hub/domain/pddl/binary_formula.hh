/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_BINARY_FORMULA_HH
#define SKDECIDE_PDDL_BINARY_FORMULA_HH

#include "formula.hh"

namespace skdecide {

    namespace pddl {

        template <typename Derived>
        class BinaryFormula : public Formula {
        public :
            typedef std::shared_ptr<BinaryFormula<Derived>> Ptr;

            BinaryFormula() {}

            BinaryFormula(const Formula::Ptr& left_formula,
                          const Formula::Ptr& right_formula)
                : _left_formula(left_formula), _right_formula(right_formula) {}
            
            BinaryFormula(const BinaryFormula<Derived>& other)
                : _left_formula(other._left_formula),
                  _right_formula(other._right_formula) {}
            
            BinaryFormula<Derived>& operator= (const BinaryFormula<Derived>& other) {
                this->_left_formula = other._left_formula;
                this->_right_formula = other._right_formula;
                return *this;
            }

            virtual ~BinaryFormula() {}

            void set_left_formula(const Formula::Ptr& formula) {
                _left_formula = formula;
            }

            const Formula::Ptr& get_left_formula() const {
                return _left_formula;
            }

            void set_right_formula(const Formula::Ptr& formula) {
                _right_formula = formula;
            }

            const Formula::Ptr& get_right_formula() const {
                return _right_formula;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " " << *_left_formula << " " << *_right_formula << ")";
                return o;
            }

        protected :
            Formula::Ptr _left_formula;
            Formula::Ptr _right_formula;
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_BINARY_FORMULA_HH
