/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_NEGATION_FORMULA_HH
#define SKDECIDE_PDDL_NEGATION_FORMULA_HH

#include "unary_formula.hh"

namespace skdecide {

    namespace pddl {

        class NegationFormula : public UnaryFormula<NegationFormula> {
        public :
            static constexpr char class_name[] = "not";

            typedef std::shared_ptr<NegationFormula> Ptr;

            NegationFormula() {}

            NegationFormula(const Formula::Ptr& formula)
                : UnaryFormula<NegationFormula>(formula) {}
            
            NegationFormula(const NegationFormula& other)
                : UnaryFormula<NegationFormula>(other) {}
            
            NegationFormula& operator= (const NegationFormula& other) {
                dynamic_cast<UnaryFormula<NegationFormula>&>(*this) = other;
                return *this;
            }

            virtual ~NegationFormula() {}
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_NEGATION_FORMULA_HH
