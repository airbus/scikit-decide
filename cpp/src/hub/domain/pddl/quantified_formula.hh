/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef AIRLAPS_PDDL_QUANTIFIED_FORMULA_HH
#define AIRLAPS_PDDL_QUANTIFIED_FORMULA_HH

#include "formula.hh"
#include "variable_container.hh"

namespace airlaps {

    namespace pddl {

        template <typename Derived>
        class QuantifiedFormula : public Formula,
                                  public VariableContainer<Derived> {
        public :
            typedef std::shared_ptr<QuantifiedFormula<Derived>> Ptr;
            typedef typename VariableContainer<Derived>::VariablePtr VariablePtr;
            typedef typename VariableContainer<Derived>::VariableVector VariableVector;

            QuantifiedFormula() {}

            QuantifiedFormula(const Formula::Ptr& formula,
                              const VariableContainer<Derived>& variables)
                : VariableContainer<Derived>(variables),
                  _formula(formula) {}
            
            QuantifiedFormula(const QuantifiedFormula& other)
                : VariableContainer<Derived>(other),
                  _formula(other._formula) {}
            
            QuantifiedFormula& operator= (const QuantifiedFormula& other) {
                dynamic_cast<VariableContainer<Derived>&>(*this) = other;
                this->_formula = other._formula;
                return *this;
            }

            virtual ~QuantifiedFormula() {}

            QuantifiedFormula& set_formula(const Formula::Ptr& formula) {
                _formula = formula;
                return *this;
            }

            const Formula::Ptr& get_formula() const {
                return _formula;
            }

            virtual std::ostream& print(std::ostream& o) const {
                o << "(" << Derived::class_name << " (";
                for (const auto& v : this->get_variables()) {
                    o << " " << *v;
                }
                o << ") " << *_formula << ")";
                return o;
            }
        
        private :
            Formula::Ptr _formula;
        };


        class UniversalFormula : public QuantifiedFormula<UniversalFormula> {
        public :
            static constexpr char class_name[] = "forall";

            typedef std::shared_ptr<UniversalFormula> Ptr;
            typedef QuantifiedFormula<UniversalFormula> VariablePtr;
            typedef QuantifiedFormula<UniversalFormula> VariableVector;

            UniversalFormula() {}

            UniversalFormula(const Formula::Ptr& formula,
                             const VariableContainer<UniversalFormula>& variables)
                : QuantifiedFormula<UniversalFormula>(formula, variables) {}
            
            UniversalFormula(const UniversalFormula& other)
                : QuantifiedFormula<UniversalFormula>(other) {}
            
            UniversalFormula& operator= (const UniversalFormula& other) {
                dynamic_cast<QuantifiedFormula<UniversalFormula>&>(*this) = other;
                return *this;
            }
        };


        class ExistentialFormula : public QuantifiedFormula<ExistentialFormula> {
        public :
            static constexpr char class_name[] = "exists";

            typedef std::shared_ptr<ExistentialFormula> Ptr;
            typedef QuantifiedFormula<ExistentialFormula> VariablePtr;
            typedef QuantifiedFormula<ExistentialFormula> VariableVector;

            ExistentialFormula() {}

            ExistentialFormula(const Formula::Ptr& formula,
                               const VariableContainer<ExistentialFormula>& variables)
                : QuantifiedFormula<ExistentialFormula>(formula, variables) {}
            
            ExistentialFormula(const ExistentialFormula& other)
                : QuantifiedFormula<ExistentialFormula>(other) {}
            
            ExistentialFormula& operator= (const ExistentialFormula& other) {
                dynamic_cast<QuantifiedFormula<ExistentialFormula>&>(*this) = other;
                return *this;
            }

            virtual ~ExistentialFormula() {}
        };

    } // namespace pddl

} // namespace airlaps

#endif // AIRLAPS_PDDL_QUANTIFIED_FORMULA_HH