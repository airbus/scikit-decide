/* Copyright (c) AIRBUS and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef SKDECIDE_PDDL_IMPLY_FORMULA_HH
#define SKDECIDE_PDDL_IMPLY_FORMULA_HH

#include "binary_formula.hh"

namespace skdecide {

    namespace pddl {

        class ImplyFormula : public BinaryFormula<ImplyFormula> {
        public :
            static constexpr char class_name[] = "imply";

            typedef std::shared_ptr<ImplyFormula> Ptr;

            ImplyFormula() {}

            ImplyFormula(const Formula::Ptr& left_formula,
                         const Formula::Ptr& right_formula)
                : BinaryFormula<ImplyFormula>(left_formula, right_formula) {}
            
            ImplyFormula(const ImplyFormula& other)
                : BinaryFormula<ImplyFormula>(other) {}
            
            ImplyFormula& operator= (const ImplyFormula& other) {
                dynamic_cast<BinaryFormula<ImplyFormula>&>(*this) = other;
                return *this;
            }

            virtual ~ImplyFormula() {}
        };

    } // namespace pddl

} // namespace skdecide

#endif // SKDECIDE_PDDL_IMPLY_FORMULA_HH
